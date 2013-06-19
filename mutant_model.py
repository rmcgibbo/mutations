"""
MSM Bayesian Mutant Model.
"""
##############################################################################
# Imports
##############################################################################

import time
import functools

import numpy as np
import pymc
import scipy.special

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


class MutantModel(object):
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


if __name__ == '__main__':
    base_counts = np.array([100, 90, 80, 70, 10, 1], dtype=float)
    observed_counts = np.array([4, 15, 10, 1, 1, 1], dtype=float) * 1
    #base_counts = 1000*np.ones_like(observed_counts)
    #base_counts = np.ones_like(observed_counts)

    mm = MutantModel(observed_counts=observed_counts, base_counts=base_counts,
                     alpha=1, beta=1)
    mm.sample(iter=20000, burn=1000, thin=100, progress_bar=False)
    print 'E[ig]', mm.expected_information_gain()


    print 'MLE', observed_counts / np.sum(observed_counts)
    print 'MAP', mm.probs.value
    print 'q  ', mm.q.value




# mle = observed_counts / np.sum(observed_counts)
#
# mcmc = pymc.MCMC(mm)
# mcmc.sample(iter=10000, burn=1000, thin=100)
# print
#
# try:
#     print 'q: %s +/- %s' % (np.mean(mcmc.trace('q')[:]),
#                             np.std(mcmc.trace('q')[:]))
# except KeyError:
#     print 'no q'
#
# for i in range(6):
#     p = mcmc.trace('probs')[:,0,i]
#     print 'p_%d: %f +/- %f     (mle=%f)' % (i, np.mean(p), np.std(p), mle[i])
#
# print expected_information_gain(mcmc.trace('probs')[:,0,:],
#                           mcmc.trace('psuedocounts')[:] + observed_counts)
#
