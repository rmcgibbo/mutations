"""
Some work-in-progress to try to come up with an example transition
matrix and mutant to apply this procedure to.
"""

import os
import sys
import numpy as np
import scipy.sparse

from msmbuilder import io, msm_analysis, MSMLib
from bayesmutant import MutantSampler

P = np.array([
  [0.8,   0.07,  0.13,  0.0,   0.0,   0.0,  0.0,  0.0,   0.0,],
  [0.10,  0.75,  0.10,  0.0,   0.0,   0.0,  0.0,  0.05,  0.0,],
  [0.08,  0.08,  0.8,   0.04,  0.0,   0.0,  0.0,  0.0,   0.0,],
  [0.0,   0.0,   0.02,  0.78,  0.1,  0.1,  0.0,  0.0,  0.0,],
  [0.0,   0.0,   0.0,   0.2,   0.6,   0.2,  0.0,  0.0,  0.0,],
  [0.0,   0.0,   0.0,   0.15,  0.2,   0.6,  0.05,  0.0,  0.0,],
  [0.0,   0.0,   0.0,   0.0,   0.0,   0.05,  0.75,  0.1,  0.1,],
  [0.0,   0.0,   0.0,   0.0,   0.0,   0.0,  0.0,  0.9,  0.1,],
  [0.0,   0.0,   0.0,   0.0,   0.0,   0.0,  0.1,  0.1,  0.8,],
])


mutant_transition_matrix = P + 0.2*scipy.sparse.rand(P.shape[0], P.shape[1], density=0.1).todense()
mutant_transition_matrix /= np.sum(mutant_transition_matrix, axis=1)

trajectory =  np.array(msm_analysis.sample(P, 0, 5000))
base_counts = MSMLib.get_counts_from_traj(trajectory).todense()


print 'base counts'
print base_counts

ms = MutantSampler(base_counts, mutant_transition_matrix)
ms.step(5000)

print 'observed counts'
print ms.counts

io.saveh('sampling.h5', base_counts=base_counts, samples=ms.samples,
                        observed_counts=ms.counts, scores=ms.scores,
                        transition_matrix=mutant_transition_matrix)
