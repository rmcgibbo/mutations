import pymc
import numpy as np


mu_0 = 0.0
tau_0 = 1.0
a_0 = 1.0
b_0 = 1.0
phi_0 = 1.0
psi_0 = 0.001
sigma = 1.0

n_systems = 2
n_states = 3
n_edges = (n_states * (n_states + 1)) / 2
sequence_length = 10

sequences = np.array([[0.2, 0.1, 0.2, 3.3, 3.1, 3.2, 5.0, 5.1, 5.0, 4.9],
                      [0.2, 0.1, 0.2, 3.3, 3.1, 3.2, 5.0, 5.1, 5.0, 4.9]])


theta_star = pymc.Normal('theta_star', mu_0, 1.0/tau_0, size=n_states)
# alpha_star = pymc.Gamma('alpha_star', a_0, b_0, size=n_edges)
beta_star = pymc.Gamma('beta_star', a_0, b_0, size=n_states * n_states)

theta = []
beta = []
states = []
transmat = []
obs = []

for j in range(n_systems):
    theta.append(
        pymc.Normal('theta_%d' % j, theta_star, 1.0 / phi_0, size=n_states))
    # alpha.append(
    #     pymc.Normal('alpha_%d' % j, alpha_star, 1.0/psi_0, size=n_edges))
    beta.append(
        pymc.Normal('beta_%d' % j, beta_star, 1.0 / psi_0, size=n_states*n_states))

    transmat = pymc.ListContainer(
        [pymc.Dirichlet('transmat_%d_%d' % (j, k), beta[j][k : k+n_states]) for k in range(n_states)])
    print transmat

    states.append([])
    obs.append([])
    states[j].append(pymc.DiscreteUniform('state_%d_0' % j, 0, n_states-1))
    for t in range(1, sequence_length):
        @pymc.dtrm
        def transrow(i=states[j][t-1]):
            row = np.zeros(n_states)
            row[0:n_states-1] = transmat[i].value
            row[n_states-1]  = 1-np.sum(row)
            return row

        states[j].append(
            pymc.Categorical('state_%d_%d' % (j, t), transrow))

    for t in range(sequence_length):
        obs[j].append(pymc.Normal(
            'obs_%d_%d' % (j, t), theta[j][states[j][t]], 1.0/sigma,
            observed=True, value=sequences[j, t]))

model = pymc.MAP([theta, beta, states, transmat, obs])
model.fit()
import IPython as ip
ip.embed()