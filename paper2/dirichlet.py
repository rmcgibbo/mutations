import itertools
import numpy as np


def dirichletprocess(alpha, base, seed=None):
    """Iterator of samples from a Dirichlet Process using the Polya urn scheme
    
    Parameters
    ----------
    alpha : float
        Concentration parameter, alpha > 0. Larger values of alpha will cause
        less repitition in the output sequence.
    base : callable
        The base measure.
    seed : {int, None}
        A random seed.
    """
    random = np.random.RandomState(seed)

    # new is the sentinel object indcating that we need to draw a new atom
    # from the base measure (with unnormalized weight alpha)
    new = object()
    atoms = [new]
    weights = [alpha]
    normalization = alpha

    for i in itertools.count():
        cs = np.cumsum(weights) / normalization
        idx = sum(cs < random.rand())
        atom = atoms[idx]
        if atom is new:
            atom = base()
            atoms.append(atom)
            weights.append(1.0)
        else:
            weights[idx] += 1.0

        normalization += 1.0
        yield atom

if __name__ == '__main__':
    for e in itertools.islice(dirichletprocess(1, np.random.randn), 10):
        print e
