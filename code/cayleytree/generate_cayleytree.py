"""
Code to build a Cayley tree graph with networkx and save a transition matrix
to disk.
"""
##############################################################################
# Imports
##############################################################################

import os

import numpy as np
from msmbuilder import msm_analysis
import scipy.linalg
import scipy.interpolate
import networkx as nx
import matplotlib.pyplot as pp
import matplotlib.gridspec as gridspec
from matplotlib.colorbar import Colorbar

outfn = 'cayleytree2.dat'
timestep = 0.5
temperature = 1

##############################################################################
# Code
##############################################################################

g = nx.Graph()


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


arms = np.array([np.nan, 0, 1, 2, 0, 0, 1, 1, 2, 2, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
wildtype_stability = np.array([0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], dtype=float)

mutant_stability = np.copy(wildtype_stability)
mutant_stability[arms == 0] += 0.5

print 'wildtype stability\n', wildtype_stability
print 'mutant stability\n', mutant_stability


# the two rate matrices
wildtype_rate = np.zeros((g.number_of_nodes(), g.number_of_nodes()))
mutant_rate = np.zeros_like(wildtype_rate)

for i, j in g.edges():
    barrier = g.edge[i][j]['barrier']
    wildtype_rate[i, j] = np.exp(-(barrier - wildtype_stability[i]) / temperature)
    wildtype_rate[j, i] = np.exp(-(barrier - wildtype_stability[j]) / temperature)
    
    mutant_rate[i, j] = np.exp(-(barrier - mutant_stability[i]) / temperature)
    mutant_rate[j, i] = np.exp(-(barrier - mutant_stability[j]) / temperature)
wildtype_rate -= np.diag(np.sum(wildtype_rate, axis=1))
mutant_rate -= np.diag(np.sum(mutant_rate, axis=1))


wildtype_tprob = scipy.linalg.expm(wildtype_rate * timestep)
mutant_tprob = scipy.linalg.expm(mutant_rate * timestep)


print 'wildtype'
print msm_analysis.get_eigenvectors(wildtype_tprob, n_eigs=2)[1][:, 0]
print 'mutant'
print msm_analysis.get_eigenvectors(mutant_tprob, n_eigs=2)[1][:, 0]

#pos = nx.spring_layout(g, iterations=1000)
#nx.draw(g, pos=pos)

wildtype = scipy.interpolate.PiecewisePolynomial(xi=[0, 0.5, 1, 1.5, 2, 2.5, 3],
    yi=[[0, 0,], [2, 0], [1, 0], [3, 0,], [2, 0], [4, 0], [3, 0]])
mutant = scipy.interpolate.PiecewisePolynomial(xi=[0, 0.5, 1, 1.5, 2, 2.5, 3],
    yi=[[0, 0,], [2, 0], [1.5, 0], [3, 0,], [2.5, 0], [4, 0], [3.5, 0]])
x = np.linspace(-0.25, 3.4, 100)

pp.plot(x, wildtype(x), color='b', linewidth=3, alpha=0.6, label='wild type')
pp.plot(x, mutant(x), color='r', linewidth=3, alpha=0.6, label='mutant')
pp.legend()
pp.xlabel('Graph Radius')
pp.xticks([0, 1, 2, 3])
pp.ylabel('Energy (kT)')
pp.ylim(-0.5, 6.5)
pp.show()

# gs = gridspec.GridSpec(1, 25)
# ax0 = pp.subplot(gs[0, 0:10])
# ax1 = pp.subplot(gs[0, 12:22])
# cb_ax = pp.subplot(gs[0,-1])
# 
# ax0.set_title('wild type')
# im = ax0.imshow(wildtype_tprob, cmap='gist_earth', interpolation='none')
# 
# ax1.set_title('mutant')
# im = ax1.imshow(mutant_tprob, cmap='gist_earth', interpolation='none')
# 
# cb = Colorbar(ax = cb_ax, mappable = im) # use the raw colorbar constructor
# cb.set_label('Transition Probability')
# cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])

# pp.savefig('fig.png')
# os.system('open fig.png')
