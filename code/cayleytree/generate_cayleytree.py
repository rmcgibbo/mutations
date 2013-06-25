"""
Code to build a Cayley tree graph with networkx and save a transition matrix
to disk.
"""
import numpy as np
import networkx as nx
import matplotlib.pyplot as pp

g = nx.Graph()
g.add_edge(0, 1)
g.add_edge(0, 2)
g.add_edge(0, 3)

def split(root, n_max, n=0):
    if n >= n_max:
        return

    left, right = len(g.nodes()), len(g.nodes()) + 1
    g.add_edge(root, left)
    g.add_edge(root, right)
    split(left, n_max, n+1)
    split(right, n_max, n+1)

split(1, 3)
split(2, 3)
split(3, 3)

#nx.draw_spring(g)
adj = np.asarray(nx.adjacency_matrix(g), dtype=float)
adj += np.diag(np.ones(len(adj)))
transition = (adj.T / np.sum(adj, axis=1)).T

np.savetxt('cayleytree3.dat', transition)
