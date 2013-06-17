"""
Some work-in-progress to try to come up with an example transition
matrix and mutant to apply this procedure to.
"""

import os
import sys
import tables

import numpy as np
import scipy.sparse
import networkx as nx
import matplotlib.pyplot as pp

from msmbuilder import msm_analysis, MSMLib, io
from mutant_model import MutantModel

P = np.array([
  [0.5,  0.3,  0.2,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,],
  [0.3,  0.4,  0.2,  0.0,  0.0,  0.0,  0.0,  0.1,  0.0,],
  [0.2,  0.1,  0.5,  0.2,  0.0,  0.0,  0.0,  0.0,  0.0,],
  [0.0,  0.0,  0.1,  0.5,  0.2,  0.2,  0.0,  0.0,  0.0,],
  [0.0,  0.0,  0.0,  0.2,  0.5,  0.3,  0.0,  0.0,  0.0,],
  [0.0,  0.0,  0.0,  0.2,  0.2,  0.5,  0.1,  0.0,  0.0,],
  [0.0,  0.0,  0.0,  0.0,  0.0,  0.1,  0.4,  0.4,  0.1,],
  [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.2,  0.6,  0.2,],
  [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.2,  0.2,  0.6,],
])
P_prime = P + scipy.sparse.rand(P.shape[0], P.shape[1], density=0.1).todense()
P_prime /= np.sum(P_prime, axis=1)

trajectory =  np.array(msm_analysis.sample(P, 0, 1000))
P_counts = MSMLib.get_counts_from_traj(trajectory).todense()

def plot(P, P_prime):
    pp.subplot(2, 2, 1)
    pp.title('P')
    pp.matshow(P, False)
    pp.colorbar()
    pp.subplot(2, 2, 2)
    pp.title('Mutant P\'')
    pp.matshow(P_prime, False)
    pp.colorbar()


class MutantSampler(object):
    def __init__(self, base_counts, tmat):
        self.base_counts = np.asarray(base_counts, dtype=np.float)
        self.observed_counts = np.zeros_like(self.base_counts, dtype=np.float)
        self.tmat = tmat
        self.n_states = base_counts.shape[0]
        
        # an order of the counts that have been sampled
        # 
        self._counts = []
        self._igs = []
    
    def choose_state(self):
        igs = []
        for i in range(self.n_states):
            # print '%d/%d' % (i, self.n_states)
            model = MutantModel(self.observed_counts[i], self.base_counts[i],
                                alpha=1, beta=1)
            model.sample(iter=11000, burn=1000, thin=100, progress_bar=False)
            ig, _ = model.expected_information_gain(n_perturb_samples=500)
            igs.append(ig)


        self._igs.append(igs)
        print 'observed counts'
        print self.observed_counts
        return np.argmax(igs)

    def sample(self, from_state):
        cs = np.cumsum(self.tmat[from_state])
        to_state = np.sum(cs < np.random.rand())

        self.observed_counts[from_state, to_state] += 1
        
        self._counts.append((from_state, to_state))
        
        print 'sampling: %s -> %s' % (from_state, to_state)
        return from_state, to_state


if __name__ == '__main__':
    print 'base counts'
    print P_counts
    
    ms = MutantSampler(P_counts, P_prime)
    
    for i in range(1000):
        idx = ms.choose_state()
        ms.sample(idx)
    
    io.saveh('sampling.h5', base_counts=P_counts, samples=np.array(ms._counts),
                             observed_counts=ms.observed_counts,
                             igs=np.array(ms._igs))
    
# print P_prime[0]
# for i in range(100):
#     print ms.sample(0)
# pp.subplot(2, 2, 4)
# pp.title('MLE Transition Matrix, P')
# pp.matshow((1.0*counts_P) / np.sum(counts_P, axis=1), False)
# pp.colorbar()
# pp.savefig('graph.png')
# pp.show()
