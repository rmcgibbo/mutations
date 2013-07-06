import os
import sys
import numpy as np
from simtk.openmm import app
from simtk import openmm as mm
from simtk import unit

input = sys.argv[1]
output = os.path.splitext(input)[0] + '.minimized' + os.path.splitext(input)[1]

pdb = app.PDBFile(input)
forcefield = app.ForceField('amber99sbildn.xml', 'amber99_obc.xml')
system = forcefield.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff, 
                                 nonbondedCutoff=1.0*unit.nanometers,
                                 rigidWater=True)
integrator = mm.LangevinIntegrator(300*unit.kelvin, 1.0/unit.picoseconds, 
                                   2.0*unit.femtoseconds)
platform = mm.Platform.getPlatformByName('CUDA')
properties = {'CudaPrecision': 'mixed'}
simulation = app.Simulation(pdb.topology, system, integrator, platform, 
                            properties)
simulation.context.setPositions(pdb.positions)

print('Minimizing...')
simulation.minimizeEnergy()

positions = simulation.context.getState(getPositions=True).getPositions(asNumpy=True)
center = np.mean(positions.value_in_unit(unit.angstroms), axis=0)
centered_positions = unit.Quantity(positions.value_in_unit(unit.angstroms) - center, unit.angstroms)

with open(output, 'w') as f:
    app.PDBFile.writeModel(pdb.topology, centered_positions, f)
print 'saved %s' % output    

