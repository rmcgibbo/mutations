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

    def expected_information_gain(self, n_perturb_samples=1000):
        """Calculate the expected information gain that would come from
        observing a single new count

        Parameters
        ----------
        n_perturb_samples : int
            Number of times to simulate sampling the single new count. (In
            addition to this sampling, this method already uses the MCMC
            trace to sample from the posterior over the model parameters

        Returns
        -------
        ig : float
            The expected information gain
        std_ig : float
            The standard deviation in the information gain.
        """
        ig = expected_information_gain(self.trace('probs')[:, 0, :],
                self.trace('psuedocounts')[:] + self.observed_counts,
                n_perturb_samples=n_perturb_samples)
        return ig


def increment_counts_from_multinomial(probs):
    """Give the indices of the counts to increment from an incremental
    observation from probs.

    This operation is done "in parallel" accross many experiments. So `probs`
    and `counts`, are not the parameters for a single dirichlet/multinomial
    distribution, they're for a whole set of dirichlet and multinomials at
    once. The first ax

    Parameters
    ----------
    probs : np.ndarray, size=(n_experiments, n_objects)
        In experiment `i`, the probability for drawing object `j` was
        counts[i, j].

    The first axes corresponds naturally to different realizations of
    `counts` and `probs` from an MCMC sampler.
    """
    def sample_multinomial(probs):
        cs = np.cumsum(probs, axis=1)
        idx = np.sum(cs < np.random.rand(*cs.shape), axis=1)
        return np.arange(len(probs)), idx

    row, col = sample_multinomial(probs)

    return row, col


def dirichlet_kl_divergence(alpha_from, alpha_to):
    """Calculate the K-L divergence from a Dirichlet distribution parameterized
    by `alpha_from` to a Dirichlet distribution parameterized by `alpha_to`

    Parameters
    ----------
    alpha_from : np.ndarray, size=(n_states)
        Concentration parameters for the first distribution
    alpha_to : np.ndarray, size=(n_states)
        Concentration parameters for the second distribution

    Returns
    -------
    kl : float
        The K-L divergence

    Note
    ----
    alpha_from and alpha_to can also be 2d, in which case the output will
    be 1d, and the results will be computed by just running the calculation
    on each row independently, as in

    kl = []
    for i in range(n_samples):
        kl.append(dirichlet_kl_divergence(alpha_from_2d[i],
                                          alpha_to_2d[i]))
    """
    alpha_from = np.asarray(alpha_from)
    alpha_to = np.asarray(alpha_to)

    # upconvert to 1 x N arrays if they supply 1d arrays
    if alpha_from.ndim == 1:
        alpha_from = alpha_from[np.newaxis, :]
    if alpha_to.ndim == 1:
        alpha_to = alpha_to[np.newaxis, :]

    assert alpha_from.shape == alpha_to.shape
    assert alpha_from.ndim == 2

    alpha_from_total = np.sum(alpha_from, axis=1)
    alpha_to_total = np.sum(alpha_to, axis=1)

    kl = (scipy.special.gammaln(alpha_from_total) -
          scipy.special.gammaln(alpha_to_total)) + \
         np.sum(scipy.special.gammaln(alpha_to) -
                scipy.special.gammaln(alpha_from), axis=1) + \
         np.sum((alpha_from - alpha_to) *
                (scipy.special.psi(alpha_from) -
                 scipy.special.psi(alpha_from_total)[:, np.newaxis]), axis=1)

    return kl

#@_timing
def expected_information_gain(probs, alphas, n_perturb_samples):
    """Compute the expected information gain that would come from observing
    a single new observation from a multinomial distribution

    Parameters
    ----------
    probs : np.ndarray, shape=(n_samples, n_objects)
        Each row is a setting of the multinomial parameters. It is assume that
        each row is an IID sample from the Dirichlet distribution.
    alpha : np.ndarray, shape=(n_samples, n_objects)
        Each row is a setting of the Dirichlet parameters. It is assume that
        each row is an IID sample from the posterior distribution of the
        Dirichlet parameters (i.e. these may be uncertain too). If the aphas
        are all fixed and only the `probs` are RVs, then just make all of
        the rows the same.
    """
    probs = np.asarray(probs)
    alphas = np.asarray(alphas)

    n_samples, n_states = probs.shape
    assert alphas.shape == probs.shape
    assert np.all(np.sum(probs, axis=1) == np.ones(n_samples))
    perturbed_alphas = np.copy(alphas)

    values = []

    for i in range(n_perturb_samples):
        row, col = increment_counts_from_multinomial(probs)
        perturbed_alphas[row, col] += 1.0
        values.append(dirichlet_kl_divergence(alphas, perturbed_alphas))
        perturbed_alphas[row, col] -= 1.0

    return np.mean(values), np.std(values)



if __name__ == '__main__':
    base_counts = np.array([100, 90, 80, 70, 10, 1], dtype=float)
    observed_counts = np.array([4, 15, 10, 1, 1, 1], dtype=float) * 1
    #base_counts = 1000*np.ones_like(observed_counts)
    #base_counts = np.ones_like(observed_counts)

    mm = MutantModel(observed_counts=observed_counts, base_counts=base_counts,
                     alpha=1, beta=1)
    M = pymc.MAP(mm)
    M.fit()

    print 'MLE', observed_counts / np.sum(observed_counts)
    print 'MAP', mm.probs.value
    print mm.q.value

    #mm.sample(iter=20000, burn=1000, thin=100, progress_bar=False)
    #print mm.expected_information_gain()



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
