###############################################################################
# Imports
###############################################################################

import numpy as np
from msmbuilder.msm_analysis import is_transition_matrix

from bayesmutant.mutant_model import MutantModel

###############################################################################
# Classes
###############################################################################


class MutantSampler(object):
    def __init__(self, base_counts, verbose=True):
        """Create the sampler
        
        Parameters
        ----------
        base_counts : np.ndarray, shape=(n_states, n_states)
            counts in the base protein, used as a prior
        transition_matrix : np.ndarray, shape=(n_states, n_states)
            transition matrix in the mutant protein. this is what
            we'll sample from.
        """
        self.base_counts = np.asarray(base_counts, dtype=np.float)
        self._observed_counts = np.zeros_like(self.base_counts, dtype=np.float)
        self.n_states = base_counts.shape[0]
        self._verbose = verbose

        # an record of the order that new transitions were attempted and observed
        self._counts = []
        self._scores = []

        # current estimate of the effective number of counts, incorporating
        # both the actual observed counts and the q-weighted prior counts
        self._eff_counts = []

    @property
    def scores(self):
        """Score of each state at each step, that determines which state was picked
        at each step
        
        Returns
        -------
        scores : np.ndarray, shape(n_samples, n_states)
            The score for each state at each step, the state that we sampled
            from at each step was np.argmax(scores, shape=1)
        """
        return np.array(self._scores)

    @property
    def samples(self):
        """The transition that was sampled at each step

        Returns
        -------
        samples : np.ndarray, shape=(n_samples, 2)
        """
        return np.array(self._counts)

    @property
    def counts(self):
        """The observed counts matrix

        Returns
        -------
        counts : np.ndarray, shape=(n_states, n_states)
        """
        return self._observed_counts

    def eff_counts(self):
        """The effective numner of counts, at each iteration
        during the sampling

        Returns
        -------
        eff_counts : np.ndarray, shape=(n_samples, n_states, n_states)
        """
        return self._eff_counts

    def choose_state(self):
        """Select a new state to sample from

        This is done using the "MutantModel" backend, which is trying
        to optimize the expected information gain. If we get an alternative
        strategy, we can try to make this plug into multiple backends or
        something.

        Returns
        -------
        index : int
            The index of the state expected to be the most productive
            to simulate from for the next step
        """
        model = MutantModel(self._observed_counts, self.base_counts,
                            alpha=1, beta=1)
        model.sample(iter=50000, burn=1000, thin=100, progress_bar=False)
        igs = model.expected_information_gain()

        self._eff_counts.append(model.eff_counts())
        self._scores.append(igs)

        if self._verbose:
            print 'igs ', [float('%.3f' % f) for f in igs]

        return np.argmax(igs)


    def step(self, n_steps):
        """Run a bunch of steps of simulation
        """
        for i in range(n_steps):
            idx = self.choose_state()
            self.sample(idx)
