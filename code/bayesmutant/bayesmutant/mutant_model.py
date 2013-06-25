"""
MSM Bayesian Mutant Model.
"""
##############################################################################
# Imports
##############################################################################

import sys
import time
import functools

import numpy as np
import pymc
import scipy.special
from scipy.special import gammaln

__all__ = ['MutantModel']


##############################################################################
# Utilities
##############################################################################

def _timing(func):
    @functools.wraps(func)
    def wrapper(*arg,**kw):
        '''source: http://www.daniweb.com/code/snippet368.html'''
        t1 = time.time()
        res = func(*arg,**kw)
        t2 = time.time()
        print '%s() timing: %fs' % (func.func_name, (t2-t1))
        return res
    return wrapper

##############################################################################
# Classes
##############################################################################


class BaseModel(object):
    "Base class for models"
    def __init__(self):
        self._sampler = pymc.MCMC(self)
    
    def sample(self, iter, burn=0, thin=1, tune_interval=1000,
               tune_throughout=True, save_interval=None, burn_till_tuned=False,
               stop_tuning_after=False, verbose=0, progress_bar=True):
        """Sample the model posterior using MCMC. This is simply a convenience
        function on top of pymc.MCMC.sample for our model. Refer to the
        documentation on that method for a description of the parameters.
        """
        self._sampler.sample(iter, burn, thin, tune_interval, tune_throughout,
                            save_interval, burn_till_tuned, stop_tuning_after,
                            verbose, progress_bar)

    def trace(self, name):
        """Get the posterior samples. Convenience to `pymc.MCMC.trace`.

        The allowable `name`s are {'probs', '_probs', 'psuedocounts', and 'q'}
        """
        return self._sampler.trace(name)


class MutantModel(BaseModel):
    def __init__(self, observed_counts, base_counts, alpha=0.5, beta=0.5):
        """Probabilistic model for the observed transition counts in a mutant
        protein
        
        Parameters
        ----------
        observed_counts : np.ndarray, shape=(n_rows, n_states)
            The number of observed transitions from state i to state j, in
            observed_counts[i, j].
        base_counts : np.ndarray, shape=(n_rows, n_states)
            The number of observed transitions in the BASE protein, of which
            this protein is a mutant.
        alpha, beta : float
            Hyperparameters for "q", the influence that the base protein has
            on our protein.
        """
        observed_counts = np.asarray(observed_counts, dtype=float)
        base_counts = np.asarray(base_counts, dtype=float)
        assert base_counts.shape == observed_counts.shape
        assert base_counts.ndim == 2
        n_rows, n_states = base_counts.shape
        
        q = pymc.Beta(name='q', alpha=alpha, beta=beta, size=n_rows,
                      value=0.5*np.ones(n_rows))
        jeffrys_counts = 0.5*np.ones_like(base_counts)
                
        @pymc.deterministic
        def psuedocounts(q=q):
            assert q.shape == (n_rows,)
            assert base_counts.shape == (n_rows, n_states)
            # want to broadcase accross the rows, so that each row i gets
            # multiplied by q[i]
            return (q*base_counts.T).T + jeffrys_counts

        @pymc.potential
        def dm_log_likelihood(alpha=psuedocounts):
            "log likelihood of dirichlet-multinomial distribution"
            ll = (gammaln(np.sum(alpha, axis=1)) - gammaln(np.sum(alpha + observed_counts, axis=1)) +
                   np.sum(gammaln(alpha + observed_counts) - gammaln(alpha), axis=1))
            assert ll.shape == (n_rows,)
            return np.sum(ll)
        
        # model
        self.q = q
        self.psuedocounts = psuedocounts
        self.dm_log_likelihood = dm_log_likelihood
        self.observed_counts = observed_counts
        super(MutantModel, self).__init__()

    def expected_information_gain(self):
        """Calculate the expected information gain that would come from
        observing a single new count

        Returns
        -------
        ig : np.ndarray, shape=(n_rows,) dtype=float
            The expected information gain from starting a simulation in each
            state.
        """
        # these are the posterior alphas
        alphas = self.trace('psuedocounts')[:] + self.observed_counts
        n_samples = len(alphas)
        alphas_total = np.sum(alphas, axis=2)
        
        igs = (scipy.special.psi(alphas_total) - np.log(alphas_total) +
               np.sum(alphas*(np.log(alphas) - scipy.special.psi(alphas)), axis=2) / alphas_total)
        assert igs.shape == (n_samples, self.observed_counts.shape[0])
        return np.mean(igs, axis=0)


class _MutantModelRow(BaseModel):
    def __init__(self, observed_counts, base_counts, alpha=0.5, beta=0.5,
                 fixed_q=None):
        """Probabilistic model for the observed outbound transition
        counts from state i.

        Parameters
        ----------
        observed_counts : np.ndarray, shape=(n_states)
            The number of observed transitions to other states in THIS protein
        base_counts : np.ndarray, shape=(n_states)
            The number of observed transitions to other states in the BASE protein,
            of which this protein is a mutant.
        alpha, beta : float
            Hyperparameters for the influence that the base protein has on
            our protein.
        fixed_q : {float, None}
            Instead of setting the influence that the base protein has as a random
            variable, make it fixed at this value. Note that when fixed_q is
            specified, alpha and beta have no influence.
        """
        base_counts = np.asarray(base_counts, dtype=float)
        observed_counts = np.asarray(observed_counts, dtype=float)
        assert base_counts.shape == observed_counts.shape

        if fixed_q is None:
            q = pymc.Beta(name='q', alpha=alpha, beta=beta)
        else:
            q = fixed_q

        jeffrys_counts = 0.5*np.ones_like(base_counts)

        @pymc.deterministic
        def psuedocounts(q=q):
            return q*base_counts + jeffrys_counts

        _probs = pymc.Dirichlet('_probs', theta=psuedocounts)
        probs = pymc.CompletedDirichlet('probs', _probs)

        counts = pymc.Multinomial(name='counts', value=observed_counts,
                                  n=np.sum(observed_counts), p=_probs,
                                  observed=True)

        # set all of the random variables as attributes
        self.q = q
        self.probs = probs
        self._probs = _probs
        self.counts = counts
        self.psuedocounts = psuedocounts
        
        self.sampler = pymc.MCMC(self)
        self.observed_counts = observed_counts


    def expected_information_gain(self):
        """Calculate the expected information gain that would come from
        observing a single new count

        Returns
        -------
        ig : float
            The expected information gain
        """
        alphas = self.trace('psuedocounts')[:] + self.observed_counts
        n_samples = len(alphas)

        alphas_total = np.sum(alphas, axis=1)

        igs = (scipy.special.psi(alphas_total) - np.log(alphas_total) +
            np.sum(alphas*(np.log(alphas) - scipy.special.psi(alphas)), axis=1) / alphas_total)
        
        assert len(igs) == n_samples
        
        return np.mean(igs)

    def sample(self, iter, burn=0, thin=1, tune_interval=1000,
               tune_throughout=True, save_interval=None, burn_till_tuned=False,
               stop_tuning_after=False, verbose=0, progress_bar=True):
        """Sample the model posterior using MCMC. This is simply a convenience
        function on top of pymc.MCMC.sample for our model. Refer to the
        documentation on that method for a description of the parameters.
        """
        self.sampler.sample(iter, burn, thin, tune_interval, tune_throughout,
                            save_interval, burn_till_tuned, stop_tuning_after,
                            verbose, progress_bar)

    def trace(self, name):
        """Get the posterior samples. Convenience to `pymc.MCMC.trace`.

        The allowable `name`s are {'probs', '_probs', 'psuedocounts', and 'q'}
        """
        return self.sampler.trace(name)


if __name__ == '__main__':
    base_counts = np.random.randint(10, size=(4,4))
    observed_counts = np.random.randint(10, size=(4,4))

    mm2 = MutantModel(observed_counts=observed_counts, base_counts=base_counts,
                       alpha=1, beta=1)
    mm2.sample(iter=20000, burn=1000, thin=100, progress_bar=False)
    print 'mean(q) [mm2]', np.mean(mm2.trace('q')[:], axis=0), ' +/-', np.std(mm2.trace('q')[:], axis=0)
    print 'mm2 E[ig]', mm2.expected_information_gain()
