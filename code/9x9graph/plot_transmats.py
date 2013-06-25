##############################################################################
# Imports
##############################################################################

import os
import sys

import numpy as np
from mdtraj import io

import matplotlib.pyplot as pp
import matplotlib.gridspec as gridspec
from matplotlib.colorbar import Colorbar

base_transition_matrix = np.loadtxt('base_transition_matrix.dat')

##############################################################################
# Globals
##############################################################################

figsize=(11,4)
sampling = io.loadh('sampling.h5')
fig = pp.figure(figsize=figsize)
plot_fn = 'plots/transition_matricies.png'

##############################################################################
# Script
##############################################################################

gs = gridspec.GridSpec(1, 25)
ax0 = pp.subplot(gs[0, 0:10])
ax1 = pp.subplot(gs[0, 12:22])
cb_ax = pp.subplot(gs[0,-1])

ax0.set_title('Base System')
im = ax0.imshow(base_transition_matrix, interpolation='nearest', origin='lower',
                vmin=0, vmax=1, cmap='gist_earth')
ax0.set_xlabel('State [index]')
ax0.set_ylabel('State [index]')


ax1.set_title('Mutant')
im = ax1.imshow(sampling['transition_matrix'], interpolation='nearest',
                origin='lower', vmin=0, vmax=1, cmap='gist_earth')
ax1.set_xlabel('State [index]')
ax1.set_ylabel('State [index]')

cb = Colorbar(ax = cb_ax, mappable = im) # use the raw colorbar constructor
cb.set_label('Transition Probability')
cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])

#cb = Colorbar(ax=pp.subplot(gs[1, -1]), mappable=lower)
#cb.set_ticks(np.asarray(np.unique(trajectory), dtype=int))
#cb.set_label('state')

sampling.close()
print 'saving figure as %s' % plot_fn
pp.savefig(plot_fn)

if sys.platform == 'darwin':
    os.system('open %s' % plot_fn)