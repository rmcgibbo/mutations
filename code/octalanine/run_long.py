import sys
import os
from simtk.openmm import app
import simtk.openmm as mm
from simtk import unit
from mdtraj.reporters import HDF5Reporter

input = sys.argv[1]
output = sys.argv[2]

ts = 2.0 * unit.femtosecond
reportInterval = int(10*unit.picoseconds / ts)
time = int(1*unit.microsecond / ts)

pdb = app.PDBFile(input)
forcefield = app.ForceField('amber99sbildn.xml', 'amber99_obc.xml')

system = forcefield.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff, 
                                 constraints=app.HBonds, rigidWater=True)
integrator = mm.LangevinIntegrator(300*unit.kelvin, 1.0/unit.picoseconds, ts)
                                   
integrator.setConstraintTolerance(0.00001)

platform = mm.Platform.getPlatformByName('CUDA')
properties = {'CudaPrecision': 'mixed'}
simulation = app.Simulation(pdb.topology, system, integrator, platform, 
                            properties)
simulation.context.setPositions(pdb.positions)


simulation.context.setVelocitiesToTemperature(300*unit.kelvin)
print 'Equilibrating...'
simulation.step(int(100*unit.picoseconds / ts))

reporters = [app.StateDataReporter(sys.stdout, reportInterval,
             step=True, time=True, potentialEnergy=True, kineticEnergy=True,
             totalEnergy=True, temperature=True),
             HDF5Reporter(output, reportInterval, cell=False)]
simulation.reporters = reporters

print 'Running Production...'

simulation.step(time)
print 'Done!'
