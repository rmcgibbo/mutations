"""
Code to build a Cayley tree graph with networkx and save a transition matrix
to disk.
"""
##############################################################################
# Imports
##############################################################################

import numpy as np
from msmbuilder import msm_analysis
import scipy.linalg
import networkx as nx
import matplotlib.pyplot as pp

outfn = 'cayleytree2.dat'
timestep = 0.5
temperature = 1

##############################################################################
# Code
##############################################################################


g = nx.Graph()
stability = [0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]


g.add_edge(0, 1, barrier=2)
g.add_edge(0, 2, barrier=2)
g.add_edge(0, 3, barrier=2)

g.add_edge(1, 4, barrier=3)
g.add_edge(1, 5, barrier=3)
g.add_edge(2, 6, barrier=3)
g.add_edge(2, 7, barrier=3)
g.add_edge(3, 8, barrier=3)
g.add_edge(3, 9, barrier=3)

g.add_edge(4, 10, barrier=4)
g.add_edge(4, 11, barrier=4)
g.add_edge(5, 12, barrier=4)
g.add_edge(5, 13, barrier=4)
g.add_edge(6, 14, barrier=4)
g.add_edge(6, 15, barrier=4)
g.add_edge(7, 16, barrier=4)
g.add_edge(7, 17, barrier=4)
g.add_edge(8, 18, barrier=4)
g.add_edge(8, 19, barrier=4)
g.add_edge(9, 20, barrier=4)
g.add_edge(9, 21, barrier=4)

wt_rate = np.zeros((g.number_of_nodes(), g.number_of_nodes()))

for i, j in g.edges():
    barrier = g.edge[i][j]['barrier']
    wt_rate[i, j] = np.exp(-(barrier - stability[i]) / temperature)
    wt_rate[j, i] = np.exp(-(barrier - stability[j]) / temperature)
    print '%d -> %d: %f' % (i, j, wt_rate[i, j])
    print '%d -> %d: %f' % (j, i, wt_rate[j, i])

wt_rate -= np.diag(np.sum(wt_rate, axis=1))
wt_t = scipy.linalg.expm(wt_rate * timestep)

print msm_analysis.get_eigenvectors(wt_t, n_eigs=2)[1][:, 0]
pp.matshow(scipy.linalg.expm(wt_rate * timestep))
pp.colorbar()
pp.show()
