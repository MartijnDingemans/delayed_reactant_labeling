import numpy as np
import pandas
from scipy.integrate import solve_ivp
from delayed_reactant_labeling.predict import DRL
import matplotlib.pyplot as plt

reactions = [
    ('k1', ['cat', 'react-H10'], ['int-H10']),
    ('k-1', ['int-H10'], ['cat', 'react-H10']),
    ('k2', ['int-H10'], ['prod-H10', 'cat']),
    ('k1', ['cat', 'react-D10'], ['int-D10']),
    ('k-1', ['int-D10'], ['cat', 'react-D10']),
    ('k2', ['int-D10'], ['prod-D10', 'cat']),
]


def calc_result(rates):
    drl = DRL(rate_constants=rates, reactions=reactions,
              output_order=['react-H10', 'react-D10', 'int-H10', 'int-D10', 'prod-H10', 'prod-D10', 'cat'])
    print(drl.reference)

    time = np.linspace(-10, 0, 1001)
    y0 = np.zeros(len(drl.reference))
    y0[drl.reference['cat']] = 0.25
    y0[drl.reference['react-H10']] = 1

    result = solve_ivp(
        drl.calculate_step,
        t_span=[time[0], time[-1]],
        y0=y0,
        method='Radau',
        t_eval=time,
        jac=drl.calculate_jac)

    result_pre = result.y.T
    yl = result_pre[-1, :]
    yl[drl.reference['react-D10']] = 1

    time1 = np.linspace(0, 30, 3001)
    result = solve_ivp(
        drl.calculate_step,
        t_span=[time1[0], time1[-1]],
        y0=yl,
        method='Radau',
        t_eval=time1,
        jac=drl.calculate_jac)

    _df = pandas.DataFrame(np.concatenate([result_pre, result.y.T]), columns=drl.reference.index)
    _df['time'] = np.concatenate([time, time1])
    return _df


fig, axs = plt.subplots(2, 1, figsize=(6.4, 6.4))
ax = axs[0]

for rates, ax in zip([{'k1': 0.5, 'k-1': 0.02, 'k2': 0.18},
                      {'k1': 0.5, 'k-1': 0.18, 'k2': 0.02}],
                     axs):
    df = calc_result(rates)
    n = 0
    for col in df:
        if col == 'time':
            continue
        if col[-3:] == 'D10':
            style = '--'
        else:
            n = n + 1
            style = '-'

        ax.plot(df['time'], df[col], color=f'C{n}', linestyle=style, label=col)
    ax.set_ylim(0, 0.2)

ax.legend(loc=1, ncol=4)

fig.show()
