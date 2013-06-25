"""
Some work-in-progress to try to come up with an example transition
matrix and mutant to apply this procedure to.
"""

import os
import sys
import tables

import numpy as np
import scipy.sparse
import matplotlib.pyplot as pp

from msmbuilder import msm_analysis, MSMLib, io
from mutant_model import MutantModel

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

def plot_convergence(transition_matrix, trajectory):
    error = []
    sublengths = range(0, len(trajectory), 10)
    
    for i in sublengths:
        section = trajectory[:i]
        try:
            counts = np.asarray(MSMLib.get_counts_from_traj(section).todense(), dtype=float)
            probs = counts / np.sum(counts, axis=1)
            assert probs.shape == (9,9)
        except:
            error.append(np.nan)
            continue
        error.append(np.linalg.norm(transition_matrix - probs))
    

    print counts
    print probs
    print transition_matrix
    
    pp.plot(sublengths, error)
    pp.ylim(0,1)

def plot_transition_matrices_and_traj(P, P_prime, trajectory):
    """Makes a plot with two imshow() transition matricies on top, sharing a
    colorbar, and then one trajectory plotted below. The layout is like:

    ####  ####  #
    ####  ####  #
    ####  ####  #
    
    #############
    #############
    #############
    
    """
    fig = pp.figure()
    import matplotlib.gridspec as gridspec
    from matplotlib.colorbar import Colorbar
    
    gs = gridspec.GridSpec(2, 21)
    ax0 = pp.subplot(gs[0, 0:10])
    ax1 = pp.subplot(gs[0, 10:20])
    cb_ax = pp.subplot(gs[0,-1])
    ax2 = pp.subplot(gs[1, 1:-2])

    ax0.set_title('base system trans. mat.')
    im = ax0.imshow(P, interpolation='nearest', origin='lower',
                    vmin=0, vmax=1, cmap='hot')

    ax1.set_title('mutant trans. mat.')
    im = ax1.imshow(P_prime, interpolation='nearest', origin='lower',
                    vmin=0, vmax=1, cmap='hot')
    
    cb = Colorbar(ax = cb_ax, mappable = im) # use the raw colorbar constructor
    cb.set_label('trans. prob')
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])

    assert np.min(trajectory) == 0 and np.max(trajectory) == 8
    yvals = np.array([0.4, 0.5, 0.6, 0.9, 1.0, 1.1, 1.4, 1.5, 1.6])

    lower = ax2.scatter(x=np.arange(len(trajectory)), y=yvals[trajectory], c=trajectory,
                        edgecolors='none', cmap='Spectral')
    ax2.set_title('base system example traj')
    ax2.set_xlabel('time [steps]')
    ax2.yaxis.set_ticklabels([])
    ax2.set_yticks([0.5, 1.0, 1.5])
    ax2.set_ylabel('state')
    
    cb = Colorbar(ax=pp.subplot(gs[1, -1]), mappable=lower)
    cb.set_ticks(np.asarray(np.unique(trajectory), dtype=int))
    cb.set_label('state')


class MutantSampler(object):
    def __init__(self, base_counts, transition_matrix):
        self.base_counts = np.asarray(base_counts, dtype=np.float)
        self.observed_counts = np.zeros_like(self.base_counts, dtype=np.float)
        self.transition_matrix = transition_matrix
        self.n_states = base_counts.shape[0]
        
        # an record of the order that new transitions were attempted and observed
        self._counts = []
        self._igs = []
    
    def choose_state(self):
        model = MutantModel(self.observed_counts, self.base_counts,
                            alpha=1, beta=1)
        model.sample(iter=50000, burn=1000, thin=100, progress_bar=False)
        igs = model.expected_information_gain()

        self._igs.append(igs)
        print 'igs ', [float('%.3f' % f) for f in igs]

        return np.argmax(igs)

    def sample(self, from_state):
        cs = np.cumsum(self.transition_matrix[from_state])
        to_state = np.sum(cs < np.random.rand())

        self.observed_counts[from_state, to_state] += 1
        
        self._counts.append((from_state, to_state))
        
        print 'sampling: %s -> %s' % (from_state, to_state)
        return from_state, to_state
    
    def step(self, n_steps):
        for i in range(n_steps):
            idx = ms.choose_state()
            ms.sample(idx)
    

if __name__ == '__main__':
    # plot_transition_matrices_and_traj(P, P_prime, trajectory)
    # plot_convergence(P, trajectory)
    # pp.show()
    
    mutant_transition_matrix = P + 0.2*scipy.sparse.rand(P.shape[0], P.shape[1], density=0.1).todense()
    mutant_transition_matrix /= np.sum(mutant_transition_matrix, axis=1)

    trajectory =  np.array(msm_analysis.sample(P, 0, 5000))
    base_counts = MSMLib.get_counts_from_traj(trajectory).todense()

    
    print 'base counts'
    print base_counts
    
    ms = MutantSampler(base_counts, mutant_transition_matrix)
    ms.step(5000)
    
    print 'observed counts'
    print ms.observed_counts

    io.saveh('sampling.h5', base_counts=base_counts, samples=np.array(ms._counts),
                            observed_counts=ms.observed_counts, igs=np.array(ms._igs),
                            transition_matrix=mutant_transition_matrix)
