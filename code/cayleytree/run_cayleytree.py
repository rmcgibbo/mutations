import os
import sys
import numpy as np
import scipy.sparse
import matplotlib.pyplot as pp

from msmbuilder import io, msm_analysis, MSMLib
from bayesmutant import MutantSampler

wt_tprob = np.loadtxt('cayleytree_tprob_wildtype.dat')
mutant_tprob = np.loadtxt('cayleytree_tprob_mutant.dat')

trajectory =  np.array(msm_analysis.sample(wt_tprob, 0, 5000))
base_counts = MSMLib.get_counts_from_traj(trajectory).todense()

print 'base counts'
print base_counts

ms = MutantSampler(base_counts, mutant_tprob)
ms.step(10)

print 'observed counts'
print ms.counts

io.saveh('sampling.h5', base_counts=base_counts, samples=ms.samples,
                        observed_counts=ms.counts, scores=ms.scores,
                        transition_matrix=mutant_tprob)
