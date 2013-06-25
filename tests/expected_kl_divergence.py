import random
import numpy as np
import scipy.special


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
    assert alpha_from.shape == alpha_to.shape

    alpha_from_total = np.sum(alpha_from)
    alpha_to_total = np.sum(alpha_to)

    kl = (scipy.special.gammaln(alpha_from_total) -
          scipy.special.gammaln(alpha_to_total)) + \
         np.sum(scipy.special.gammaln(alpha_to) -
                scipy.special.gammaln(alpha_from)) + \
         np.sum((alpha_from - alpha_to) *
                (scipy.special.psi(alpha_from) -
                 scipy.special.psi(alpha_from_total)))

    return kl

def incremental_kl_divergence(alpha, i):
    """Calculate the 
    
    
    """
    alpha_total = np.sum(alpha)
    return (-np.log(alpha_total) + np.log(alpha[i]) + 
            scipy.special.psi(alpha_total) - scipy.special.psi(alpha[i]))


def sample_incremental_kl_divergence(alphas):
    # sample probs ~ Dir(alphas)
    probs = np.random.gamma(alphas, scale=1)
    probs /= np.sum(probs)
    
    # sample from Multi(1; probs)
    cumsum = np.cumsum(probs)
    l = np.sum(cumsum < np.random.rand())  # index of the selected value

    return incremental_kl_divergence(alphas, l)



def analytic_incremental_kl_divergence(alphas):
    alphas_total = np.sum(alphas)
    
    return (scipy.special.psi(alphas_total) - np.log(alphas_total) +
            np.sum(alphas*(np.log(alphas) - scipy.special.psi(alphas))) / alphas_total)


def test1():
    n_states = 10
    scale = 5
    
    alpha = scale*np.random.rand(n_states)
    i = np.random.randint(n_states)

    increment_vector = np.zeros(n_states)
    increment_vector[i] = 1
    
    print incremental_kl_divergence(alpha, i)
    print dirichlet_kl_divergence(alpha, alpha+increment_vector)


if __name__ == '__main__':
#    test1()
    alphas = np.asarray([1,1,2,4,1], dtype=float) * 10
    print 'alphas', alphas
    
    
    print 'analytic', analytic_incremental_kl_divergence(alphas)
    n_samples = 10000
    
    print 'sampled ', sum(sample_incremental_kl_divergence(alphas) for i in range(n_samples)) / n_samples