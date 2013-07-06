"""
Code to thread one topology onto the positions of another.
"""
###############################################################################
# Imports
###############################################################################

from itertools import ifilter
import networkx as nx
import numpy as np

import simtk.openmm as mm
from simtk.openmm import app
from simtk import unit as units

###############################################################################
# Globals
###############################################################################

__all__ = ['rethread']

###############################################################################
# Functions
###############################################################################


def topology_to_nx(topology):
    """Create a networkx graph from a Topology"""
    graph = nx.Graph()
    for atom in topology.atoms():
        graph.add_node(atom.index, element=atom.element.symbol)

    for atom1, atom2 in topology.bonds():
        graph.add_edge(atom1.index, atom2.index)

    return graph


def rethread(mutant_topology, wildtype_topology, wildtype_positions, forcefield=None):
    """Create a configuration of a mutant in the same "state"
    as a given configuration of the wildtype.

    This tries to find a subgraph of the mutant which is isomorphic with the
    wildtype graph, and uses that to map positions. The remaining positions are
    determined by minimzing the energy of the system, with constraints applied
    to the mapped atoms.

    This method only works if the number of residues in the mutant is the same
    as the number of residues in the mutant.

    Parameters
    ----------
    mutant_topology : Topology
        Topology of the mutant system
    wildtype_topology : Topology
        Topology of the wildtype system
    wildtype_positions : list
        Positions of the atoms in the wildtype system
    forcefield : openmm.ForceField
        Which forcefield to use to minimize the new atoms. Default=amber99sbildn

    Returns
    -------
    mutant_positions : list of Vec3
        Positions of the mutant atoms, as an openmm-compatible list of Vec3s.
    """
    if not len(list(wildtype_topology.residues())) == len(list(mutant_topology.residues())):
        raise ValueError('the two topologies must have the same # of residues')
    for c1, c2 in zip(mutant_topology.chains(), wildtype_topology.chains()):
        if not c1.index == c2.index:
            raise ValueError('two topologies must have the same # of chains')
    if not np.all(np.array([a.index for a in wildtype_topology.atoms()]) == np.arange(len(wildtype_positions))):
        raise ValueError('number of atoms in topology and positions dont match up')


    # remove side chain atoms in the wildtype topology for residues
    # that are different in the wildtype vs mutant
    wt_modeller = app.Modeller(wildtype_topology, wildtype_positions)
    for i, (wr, mr) in enumerate(zip(wildtype_topology.residues(),
                                     mutant_topology.residues())):
        if wr.name != mr.name:
            wt_modeller.delete([a for a in wr.atoms() if a.name not in ['C', 'O', 'N', 'CA']])
    wildtype_positions = wt_modeller.positions
    wildtype_topology = wt_modeller.topology

    g_mutant = topology_to_nx(mutant_topology)
    g_wildtype = topology_to_nx(wildtype_topology)

    gm = nx.isomorphism.GraphMatcher(g_mutant, g_wildtype)

    try:
        isomorphism = gm.subgraph_isomorphisms_iter().next()
    except StopIteration:
        raise ValueError('No subgraph isomorphism found.')

    # create array in which to put the mutant positions
    wildtype_positions = np.asarray(wildtype_positions.value_in_unit(units.nanometers))
    mutant_positions = np.empty((mutant_topology._numAtoms, 3))
    mutant_positions.fill(np.nan)

    # thread on all of the atoms that have an isomorphism in the wildtype. the remaining
    # unmatched atoms now have coordinates NaN
    for mutant_atom, wildtype_atom in isomorphism.iteritems():
        mutant_positions[mutant_atom] = wildtype_positions[wildtype_atom]
    positionless_atoms = np.where(np.isnan(mutant_positions[:, 0]))[0]
    fixed_atoms = np.where(np.logical_not(np.isnan(mutant_positions[:, 0])))[0]

    # fill in random starting positions for the mutant atoms that aren't in the isomorphism
    while len(positionless_atoms) != 0:
        for a in positionless_atoms:
            try:
                # get an atom that we're bonded to who does have a position in the graph
                n = ifilter(lambda e: e not in positionless_atoms, g_mutant.neighbors(a)).next()
            except StopIteration:
                continue

            # just set the mutant position to be "nearby"
            mutant_positions[a] = wildtype_positions[n] + np.random.randn(3)
        positionless_atoms = np.where(np.isnan(mutant_positions[:, 0]))[0]

    # create the openmm system
    if forcefield is None:
        forcefield = app.ForceField('amber99sbildn.xml')
    system = forcefield.createSystem(mutant_topology, nonbondedMethod=app.CutoffNonPeriodic,
                                     nonbondedCutoff=1.0*units.nanometers, constraints=None)

    # add position restraints to all of the atoms whose position is fixed by the
    # isomorphism. we only want to mimize the unknown atoms
    restraint = mm.CustomExternalForce("k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    restraint.addGlobalParameter('k', 100.0)
    restraint.addPerParticleParameter('x0')
    restraint.addPerParticleParameter('y0')
    restraint.addPerParticleParameter('z0')
    for atom in fixed_atoms:
        restraint.addParticle(int(atom), mutant_positions[atom].tolist())
    system.addForce(restraint)

    integrator = mm.VerletIntegrator(2.0*units.femtoseconds)
    simulation = app.Simulation(mutant_topology, system, integrator, platform=mm.Platform.getPlatformByName('Reference'))
    simulation.context.setPositions(units.Quantity(mutant_positions.tolist(), units.nanometers))

    # run the minimizer
    simulation.minimizeEnergy()
    state = simulation.context.getState(getPositions=True, getEnergy=True)

    # check for errors
    if np.isnan(state.getPotentialEnergy().value_in_unit(units.kilojoules_per_mole)):
        raise ValueError('energy is nan')

    return state.getPositions()


###############################################################################
# script
###############################################################################


def main():
    mutant = app.PDBFile('../../octalanine/structures/mutant.minimized.pdb')
    wildtype = app.PDBFile('../../octalanine/structures/octalanine.minimized.pdb')
    mutant, wildtype = wildtype, mutant

    positions = rethread(mutant.topology, wildtype.topology, wildtype.positions)

    app.PDBFile.writeModel(wildtype.topology, wildtype.positions, open('base.pdb', 'w'))
    app.PDBFile.writeModel(mutant.topology, positions, open('rethreaded.pdb', 'w'))

    print 'saved base.pdb and rethreaded.pdb.'


if __name__ == '__main__':
    main()
