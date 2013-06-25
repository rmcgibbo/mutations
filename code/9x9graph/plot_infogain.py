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
plot_fn = 'plots/information_gain.png'
n_states = 9
upto = 600
figsize = (10, 6)  # inches

# setthe default colormap
mpltools.color.cycle_cmap(n_states, cmap=cm)

# load the data
sampling = io.loadh('sampling.h5')
try:
    igs = sampling['scores']
except KeyError:
    # the old code put it under this name
    igs = sampling['igs']

igs = igs[:upto]
samples = sampling['samples'][:upto]
n_counts = len(samples)
sampling.close()

##############################################################################
# Script
##############################################################################

pp.figure(figsize=figsize)
pp.title('sampling the 9 state mutant')

##############
# Top subplot
##############
ax = pp.subplot(2, 1, 1)
df = pd.DataFrame(igs, columns=['state %d' % i for i in range(n_states)])
df.plot(legend=False, ax=ax)
ax.set_ylabel('Expected Information Gain [nats]')
ax.set_xlim(-1, n_counts)
pp.xlabel('Time [steps]')


#################
# bottom subplot
#################
ax = pp.subplot(2, 1, 2)
ax.scatter(x=np.arange(n_counts), y=samples[:, 0],
           c=mpltools.color.colors_from_cmap(n_states, cm)[samples[:, 0]],
           edgecolor='none')
ax.set_ylabel('Sampled State [index]')
ax.set_ylim(-1, n_states)
ax.set_xlim(-1, n_counts)
ax.set_yticks(range(n_states))
pp.xlabel('Time [steps]')


#pp.show()
print 'saving figure as %s' % plot_fn
pp.savefig(plot_fn)

if sys.platform == 'darwin':
    os.system('open %s' % plot_fn)
