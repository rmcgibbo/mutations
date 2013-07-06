import numpy as np

from simtk import unit
from simtk.openmm import app
import simtk.openmm as mm

from bayesmutant.mutant_rethreading import rethread
from bayesmutant.sampler import MutantSampler

import mdtraj as md

class OpenMMMutantSampler(MutantSampler):
    def __init__(self, wildtype_counts, wildtype_states_fn, wildtype_topology_fn, mutant_topology_fn, lag_time, simulation):
        """Create an OpenMM sampler

        Parameters
        ----------
        wildtype_counts : np.ndarray, shape=(n_states, n_states)
            The number of observed transition counts in the base
            protein
        wildtype_states_fn : string
            Path to a trajectory file giving the states at the
            centers of each state, for the wildtype topology.
            The states must be defined by RMSD.
        wildtype_topology_fn : string
            Path to a file giving the topology of the wildtype.
            Should be a pdb.
        mutant_topology_fn : string
            Path to a file giving the topology of the mutant.
            Should be a pdb.
        lag_time : simtk.unit.Quantity
            The msm lag time, in real time units.
        simulation : simtk.openmm.app.Simulation
            A simulation object, set up for the mutant.
        """
        self._wt_states = md.load(wildtype_states_fn)
        wt_backbone_inds = [a.index for a in self._wt_states.topology.atoms if a.name in ['C', 'CA', 'CB', 'N', 'O']]
        assert len(wt_backbone_inds) > 0
        wt_backbone = md.load(wildtype_states_fn, atom_indices=wt_backbone_inds)
        self._wt_rmsd_cache = md.rmsd_cache(wt_backbone)

        self._wt_topology = app.PDBFile(wildtype_topology_fn).topology
        self._lag_time = lag_time
        self._mutant_topology = app.PDBFile(mutant_topology_fn).topology

        self._mutant_backbone_inds = [a.index for a in self._mutant_topology.atoms() if a.name in ['C', 'CA', 'CB', 'N', 'O']]

        if not len(list(self._wt_topology.atoms())) == self._wt_states.n_atoms:
            raise ValueError('wt topology fn must have same # of states as wt state file')


        self._simulation = simulation
        super(OpenMMMutantSampler, self).__init__(wildtype_counts, verbose=True)


    def sample(self, from_state):
        """Run a single round of sampling"""
        print 'OpenMMSampler.sample(from_state=%d)' % from_state

        print 'theading...'
        mutant_positions = rethread(self._mutant_topology, self._wt_topology,
                                    self._wt_states.xyz[from_state].tolist())

        print 'setting state...'
        self._simulation.context.setPositions(mutant_positions)
        self._simulation.context.setVelocitiesToTemperature(300*unit.kelvin)

        step_size = self._simulation.context.getIntegrator().getStepSize()
        n_steps = int(self._lag_time / step_size)

        # todo: add a reporter
        print 'stepping for %d steps (%s)' % (n_steps, self._lag_time)
        self._simulation.step(n_steps)

        print 'done sampling. querying state...'
        positions = self._simulation.context.getState(getPositions=True).getPositions(asNumpy=True)
        backbone_positions = positions[self._mutant_backbone_inds].T

        # make a single-frame rmsd cache
        rmsdc = md.rmsd.RMSDCache(md.rmsd.align_array(backbone_positions[np.newaxis, :, :], 'axis'),
                                  'axis', len(self._mutant_backbone_inds))
        distances = self._wt_rmsd_cache.rmsds_to(rmsdc, 0)

        to_state = np.argmin(distances)
        print 'to_state = %d' % to_state

        print 'one more observed count for %d->%d' % (from_state, to_state)
        self._observed_counts[from_state, to_state] += 1
        self._counts.append((from_state, to_state))
        
        return from_state, to_state
        
