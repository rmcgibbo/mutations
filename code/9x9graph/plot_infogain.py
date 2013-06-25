##############################################################################
# Imports
##############################################################################

import os
import sys

from mdtraj import io

import numpy as np
import pandas as pd
import matplotlib.pyplot as pp
import mpltools.color

##############################################################################
# Globals
##############################################################################

pd.options.display.mpl_style='default'
pp.rcParams['font.family'] = 'sans-serif'
cm = 'Set1'
plot_fn = 'information_gain.png'
n_states = 9
figsize = (10, 6)  # inches

# setthe default colormap
mpltools.color.cycle_cmap(n_states, cmap=cm)

##############################################################################
# Script
##############################################################################

sampling = io.loadh('sampling.h5')
n_counts = len(sampling['samples'])
pp.figure(figsize=figsize)

pp.title('sampling the 9 state mutant')

##############
# Top subplot
##############
ax = pp.subplot(2, 1, 1)
df = pd.DataFrame(sampling['igs'], columns=['state %d' % i for i in range(n_states)])
df.plot(legend=False, ax=ax)
ax.set_ylabel('Expected Information Gain [nats]')
ax.set_xlim(-1, len(sampling['samples']))
pp.xlabel('Time [steps]')


#################
# bottom subplot
#################
ax = pp.subplot(2, 1, 2)
from_states = sampling['samples'][:, 0]
ax.scatter(x=np.arange(len(sampling['samples'])), y=from_states,
           c=mpltools.color.colors_from_cmap(n_states, cm)[from_states],
           edgecolor='none')
ax.set_ylabel('Sampled State [index]')
ax.set_ylim(-1, n_states)
ax.set_xlim(-1, n_counts)
ax.set_yticks(range(n_states))
pp.xlabel('Time [steps]')


sampling.close()
#pp.show()
print 'saving figure as %s' % plot_fn
pp.savefig(plot_fn)

if sys.platform == 'darwin':
    os.system('open %s' % plot_fn)