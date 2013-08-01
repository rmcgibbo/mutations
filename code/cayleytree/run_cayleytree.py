import os
import sys
import numpy as np
import scipy.sparse
import matplotlib.pyplot as pp

from msmbuilder import io, msm_analysis, MSMLib
from bayesmutant import SimpleMutantSampler

wt_tprob = np.loadtxt('cayleytree_tprob_wildtype.dat')
mutant_tprob = np.loadtxt('cayleytree_tprob_mutant.dat')

base_counts = np.zeros_like(wt_tprob)
for i in range(base_counts.shape[0]):
    base_counts[i] = np.random.multinomial(200, wt_tprob[i])

print 'base counts'
print base_counts

ms = SimpleMutantSampler(base_counts, mutant_tprob)
ms.step(5000)

#print ms.eff_counts()

#print 'observed counts'
#print ms.counts

io.saveh('sampling2.h5', base_counts=base_counts, samples=ms.samples,
                        observed_counts=ms.counts, scores=ms.scores,
                        effective_counts=ms.eff_counts(),
                        transition_matrix=mutant_tprob)
