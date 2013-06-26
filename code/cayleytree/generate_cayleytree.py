"""
Code to build a Cayley tree graph with networkx

Saves both a wildtype and mutant transition matrix to disk, as well as a
figure that shows what's going on
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

def build_networks():
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

    # print 'wildtype stability\n', wildtype_stability
    # print 'mutant stability\n', mutant_stability


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

    return g, arms, wildtype_tprob, mutant_tprob


def plot_energy(ax):
    wildtype = scipy.interpolate.PiecewisePolynomial(xi=[0, 0.5, 1, 1.5, 2, 2.5, 3],
        yi=[[0, 0,], [2, 0], [1, 0], [3, 0,], [2, 0], [4, 0], [3, 0]])
    mutant = scipy.interpolate.PiecewisePolynomial(xi=[0, 0.5, 1, 1.5, 2, 2.5, 3],
        yi=[[0, 0,], [2, 0], [1.5, 0], [3, 0,], [2.5, 0], [4, 0], [3.5, 0]])
    x = np.linspace(-0.25, 3.4, 100)

    ax.plot(x, wildtype(x), color='b', linewidth=3, alpha=0.6, label='wildtype arm')
    ax.plot(x, mutant(x), color='r', linewidth=3, alpha=0.6, label='mutant arm')
    ax.legend(loc=2, prop={'size': 10})
    ax.set_xlabel('Graph Radius')
    ax.set_xticks([0, 1, 2, 3])
    ax.set_ylabel('Energy (kT)')
    ax.set_ylim(-0.5, 6.5)


def polar2cart(r, theta):
    theta = theta * np.pi / 180.0
    return r*np.sin(theta), r*np.cos(theta)

def plot_network(ax, g, arms, title, colors):
    pos = nx.spring_layout(g, iterations=1000)
    pos = {
        0: [0, 0],
        1: polar2cart(1, 0),
        2: polar2cart(1, 120),
        3: polar2cart(1, 240),
    }
    for i in range(4, 10):
        pos[i] = polar2cart(2, (i-4)*60.0 - 30)
    for i in range(10, 22):
        pos[i] = polar2cart(3, (i - 10)*30 - 45)


    arm_01 = [(a,b) for (a,b) in g.edges() if arms[a] != 2 and arms[b] != 2]
    arm_2 = [(a,b) for (a,b) in g.edges() if arms[a] == 2 or arms[b] == 2]

    # plot the edges
    nx.draw_networkx_edges(g, pos, ax=ax, width=4, edge_color=colors[0], edgelist=arm_01, alpha=0.65)
    nx.draw_networkx_edges(g, pos, ax=ax, width=4, edge_color=colors[1], edgelist=arm_2, alpha=0.65)

    # plot the nodes
    xy = np.asarray([pos[v] for v in g.nodes()])
    node_collection = ax.scatter(xy[:,0], xy[:, 1], c='#A0CBE2', s=300)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)

    circle1 = pp.Circle((0,0), 1, color='k', clip_on=False, alpha=0.25, fill=False)
    circle2 = pp.Circle((0,0), 2, color='k', clip_on=False, alpha=0.25, fill=False)
    circle3 = pp.Circle((0,0), 3, color='k', clip_on=False, alpha=0.25, fill=False)

    ax.add_artist(circle1)
    ax.add_artist(circle2)
    ax.add_artist(circle3)

if __name__ == '__main__':
    pp.figure(figsize=(7*1.5, 3.1*1.5))
    g, arms, wt_tprob, mutant_tprob = build_networks()

    ax1 = pp.subplot2grid((1,5), (0,0), colspan=2)
    ax2 = pp.subplot2grid((1,5), (0,2), colspan=1)
    ax3 = pp.subplot2grid((1,5), (0,3), colspan=2)
    

    plot_network(ax1, g, arms, title='wild type network', colors=['b', 'b'])
    plot_energy(ax2)
    plot_network(ax3, g, arms, title='mutant network', colors=['b', 'r'])
    pp.tight_layout()
    
    np.savetxt('cayleytree_tprob_wildtype.dat', wt_tprob)
    np.savetxt('cayleytree_tprob_mutant.dat', mutant_tprob)
    
    pp.savefig('figures/cayley.png')
    os.system('open figures/cayley.png')