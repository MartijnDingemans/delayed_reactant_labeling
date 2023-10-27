reaction1 = ('k1', ['A'], ['B'])
reaction2 = ('k2', ['B'], ['C'])
reactions = [reaction1, reaction2]

rate_constants = {
    "k1": 0.2,
    "k2": 0.5,
}

import numpy as np
from scipy.integrate import solve_ivp
from delayed_reactant_labeling.predict_new import DRL
time = np.linspace(0, 20, 2000)
k1, k2 = rate_constants['k1'], rate_constants['k2']
A0 = 1

# examine these data points. unit of time here determines unit of time of rate constants.
drl = DRL(rate_constants=rate_constants, reactions=reactions, output_order=['A', 'B', 'C'], verbose=False)

jac = [[-k1,    0,      0],
       [k1,     -k2,    0],
       [0,      k2,     0]]
jac = np.array(jac).T
print(jac)
result = solve_ivp(drl.calculate_step, t_span=[0, 20], y0=[A0, 0, 0], method='Radau', t_eval=time, jac=jac)

import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
kinetic_A = A0 * np.exp(-k1 * time)
kinetic_B = k1 / (k2 - k1) * A0 * (np.exp(-k1 * time) - np.exp(-k2 * time))
kinetic_C = A0 * (1 - np.exp(-k1 * time) - k1 / (k2 - k1) * (np.exp(-k1 * time) - np.exp(-k2 * time)))

eA = mean_absolute_error(kinetic_A, result.y[0])
eB = mean_absolute_error(kinetic_B, result.y[1])
eC = mean_absolute_error(kinetic_C, result.y[2])

fig, ax = plt.subplots()
ax.plot(time, result.y[0] / A0, label='A')
ax.plot(time, result.y[1] / A0, label='B')
ax.plot(time, result.y[2] / A0, label='C')
ax.plot(time, kinetic_A, color='k', linestyle=':', label='algebraic')
ax.plot(time, kinetic_B, color='k', linestyle=':')
ax.plot(time, kinetic_C, color='k', linestyle=':')
ax.set_title(f'total error: {eA+eB+eC:.6e}')
ax.legend()
fig.show()
