"""
Some work-in-progress to try to come up with an example transition
matrix and mutant to apply this procedure to.
"""

import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as pp
from msmbuilder import msm_analysis, MSMLib

P = np.array([
  [0.5,  0.3,  0.2,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,],
  [0.3,  0.4,  0.2,  0.0,  0.0,  0.0,  0.0,  0.1,  0.0,],
  [0.2,  0.1,  0.5,  0.2,  0.0,  0.0,  0.0,  0.0,  0.0,],
  [0.0,  0.0,  0.1,  0.5,  0.2,  0.2,  0.0,  0.0,  0.0,],
  [0.0,  0.0,  0.0,  0.2,  0.5,  0.3,  0.0,  0.0,  0.0,],
  [0.0,  0.0,  0.0,  0.2,  0.2,  0.5,  0.1,  0.0,  0.0,],
  [0.0,  0.0,  0.0,  0.0,  0.0,  0.1,  0.4,  0.4,  0.1,],
  [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.2,  0.6,  0.2,],
  [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.2,  0.2,  0.6,],
])
P2 = np.array([np.random.multinomial(20, p) for p in P], dtype=float)
P2 /= np.sum(P2, axis=1)
assert msm_analysis.is_transition_matrix(P)
assert msm_analysis.is_transition_matrix(P2)

pp.subplot(2, 2, 1)
pp.title('P')
pp.matshow(P, False)
pp.colorbar()
pp.subplot(2, 2, 2)
pp.title('Mutant P\'')
pp.matshow(P2, False)
pp.colorbar()


trajectory =  np.array(msm_analysis.sample(P, 0, 1000))
P_counts = MSMLib.get_counts_from_traj(trajectory).todense()

pp.subplot(2, 2, 3)
pp.title('Observed Counts')
pp.matshow(P_counts, False)
pp.colorbar()



# pp.subplot(2, 2, 4)
# pp.title('MLE Transition Matrix, P')
# pp.matshow((1.0*counts_P) / np.sum(counts_P, axis=1), False)
# pp.colorbar()






pp.show()
