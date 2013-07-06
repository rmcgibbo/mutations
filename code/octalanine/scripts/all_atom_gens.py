"""
Little script to find all-atom generators from the dimensionality reduced
generators.
"""
import mdtraj as md
import numpy as np
import mdtraj.compatibility

trjs = md.load('Trajectories/trj0.lh5')
trjs.restrict_atoms(np.loadtxt('AtomIndices.dat', int))

rc = md.rmsd_cache(trjs)
gens = md.rmsd_cache(md.load('Data_RMSD2_5A/Gens.lh5'))

ks = []
for i in range(len(gens)):
    k = np.argmin(rc.rmsds_to(gens, i))
    ks.append(k)

t = md.load('Trajectories/trj0.lh5')
print 'Done'
md.Trajectory(xyz=t.xyz[ks], topology=t.topology).save('Data_RMSD2_5A/Gens.h5')
