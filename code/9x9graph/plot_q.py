from scipy.stats import gaussian_kde
from bayesmutant import MutantModel
import brewer2mpl
from mdtraj import io
import numpy as np
import matplotlib
matplotlib.rcParams['font.size'] = 16
import matplotlib.pyplot as pp


#s = io.loadh('sampling.h5')
#m = MutantModel(s['observed_counts'], s['base_counts'])
#m.sample(100000, burn=10000, thin=1000)


x = np.linspace(0,1,1000)

bmap = brewer2mpl.get_map('Set1', 'Qualitative', 9)

for i in range(9):
    pp.plot(x, gaussian_kde(m.trace('q')[:, i])(x), lw=2, label='%d' % i, color=bmap.mpl_colors[i])
    pp.fill_between(x, gaussian_kde(m.trace('q')[:, i])(x), np.zeros_like(x), lw=2, label='%d' % i, color=bmap.mpl_colors[i], alpha=0.3)

pp.xlabel(r'Transfer coefficients, $q_i$')
pp.ylabel('Posterior Probability $P(q_i)$')
pp.grid()

pp.legend(prop={'size':14})
