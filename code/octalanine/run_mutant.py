###############################################################################
# Imports
###############################################################################
# stdlib
import sys
from os.path import join, abspath, dirname

# third party
import scipy.io
import simtk.unit as units
from simtk.openmm import app
import simtk.openmm as mm
from mdtraj import io

# this package
from bayesmutant import OpenMMMutantSampler


###############################################################################
# Globals
###############################################################################

root = dirname(abspath(__file__))
wildtype_topology_fn = join(root, 'wildtype_msm/native.pdb')
mutant_topology_fn = join(root, 'structures/mutant.minimized.pdb')


###############################################################################
# Script
###############################################################################

def main():
    wildtype_states_fn = join(root, 'wildtype_msm/Data_RMSD2_5A/Gens.h5')
    wildtype_counts = scipy.io.mmread(join(root, 'wildtype_msm',
                                           'Data_RMSD2_5A/500ps_MLE/tCounts.mtx')).todense()
    lag_time = 500*units.picoseconds


    ms = OpenMMMutantSampler(wildtype_counts, wildtype_states_fn,
                    wildtype_topology_fn, mutant_topology_fn, lag_time,
                    simulation=setup_simulation())
    ms.step(2)


    io.saveh('sampling.h5', base_counts=wildtype_counts, samples=ms.samples,
                        observed_counts=ms.counts, scores=ms.scores)




def setup_simulation():
    ts = 2.0 * units.femtosecond
    pdb = app.PDBFile(mutant_topology_fn)
    reportInterval = int(20 * units.picoseconds / ts)

    forcefield = app.ForceField('amber99sbildn.xml', 'amber99_obc.xml')
    system = forcefield.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff, 
                                     constraints=app.HBonds, rigidWater=True)
    integrator = mm.LangevinIntegrator(300*units.kelvin, 1.0/units.picoseconds, ts)
    integrator.setConstraintTolerance(0.00001)
    platform = mm.Platform.getPlatformByName('CUDA')
    properties = {'CudaPrecision': 'mixed'}
    simulation = app.Simulation(pdb.topology, system, integrator, platform, 
                                properties)
    reporters = [app.StateDataReporter(sys.stdout, reportInterval,
             step=True, time=True, potentialEnergy=True, kineticEnergy=True,
             totalEnergy=True, temperature=True)]
    simulation.reporters = reporters

    return simulation

if __name__ == '__main__':
    main()
