###############################################################################
# Imports
###############################################################################

import numpy as np
from msmbuilder.msm_analysis import is_transition_matrix

from bayesmutant.sampler import MutantSampler


###############################################################################
# Classes
###############################################################################

class SimpleMutantSampler(MutantSampler):
    def sample(self, from_state):
        """Run a single round of sampling

        Parameters
        ----------
        from_state : int
            the state you want to sample from

        Returns
        -------
        transition : (from_state, to_state)
            The pair of states that define the transition. the to_state
            is chosen from the appropriate multinomial.
        """
        cs = np.cumsum(self._transition_matrix[from_state])
        to_state = np.sum(cs < np.random.rand())

        self._observed_counts[from_state, to_state] += 1
        self._counts.append((from_state, to_state))

        if self._verbose:
            print 'sampling: %s -> %s' % (from_state, to_state)
        return from_state, to_state

    def __init__(self, base_counts, transition_matrix, verbose=True):
        """Create the sampler

        Parameters
        ----------
        base_counts : np.ndarray, shape=(n_states, n_states)
            counts in the base protein, used as a prior
        transition_matrix : np.ndarray, shape=(n_states, n_states)
            transition matrix in the mutant protein. this is what
            we'll sample from.
        """
        self._transition_matrix = transition_matrix
        assert is_transition_matrix(self._transition_matrix), 'not a transition matrix'

        super(SimpleMutantSampler, self).__init__(base_counts, verbose)
