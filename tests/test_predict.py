from time import perf_counter

import numpy as np
from scipy.integrate import solve_ivp
from sklearn.metrics import mean_absolute_percentage_error

from delayed_reactant_labeling.predict import DRL

k1 = .2
k2 = .5
rate_constants = {"k1": k1, "k2": k2}
reactions_ABC = [
    ("k1", ["A"], ["B"]),
    ("k2", ["B"], ["C"]), ]


def test_jac_simple():
    drl = DRL(reactions=reactions_ABC, rate_constants=rate_constants)
    J = drl.calculate_jac(None, np.array([1, 0, 0]))
    J_expected = np.array([
        [-k1, 0, 0],
        [k1, -k2, 0],
        [0, k2, 0],
    ])
    assert np.allclose(J, J_expected)


def test_jac_simple_with_cat():
    reactions_with_cat = [
        ('k1', ['cat', 'A'], ['B'],),
        ('k2', ['B'], ['C', 'cat'],)
    ]
    cat0 = 2
    A0 = 2

    drl = DRL(reactions=reactions_with_cat, rate_constants=rate_constants, output_order=['cat', 'A', 'B', 'C'])
    J = drl.calculate_jac(None, np.array([cat0, A0, 0, 0]))

    J_expected = np.array([
        [-k1 * A0, -k1 * cat0, k2, 0],
        [-k1 * A0, -k1 * cat0, 0, 0],
        [k1 * A0, k1 * cat0, -k2, 0],
        [0, 0, k2, 0],
    ])
    assert(np.allclose(J, J_expected))


def test_model_performance():
    atol = 1e-10
    rtol = 1e-10

    A0 = 1
    time = np.linspace(0, 20, 1000)

    # algebraic solution
    kinetic_A = A0 * np.exp(-k1 * time)
    kinetic_B = k1 / (k2 - k1) * A0 * (np.exp(-k1 * time) - np.exp(-k2 * time))
    kinetic_C = A0 * (1 - np.exp(-k1 * time) - k1 / (k2 - k1) * (np.exp(-k1 * time) - np.exp(-k2 * time)))

    # make sure everything has compiled
    drl = DRL(
        rate_constants=rate_constants, reactions=reactions_ABC, output_order=['A', 'B', 'C'], verbose=False)
    _ = solve_ivp(
        drl.calculate_step, t_span=[time[0], time[-1]], y0=[A0, 0, 0], method='Radau', t_eval=time,
        jac=drl.calculate_jac, rtol=rtol, atol=atol)

    # predict new
    ti = perf_counter()
    print('\n\nDRL new, no jac')
    drl = DRL(
        rate_constants=rate_constants, reactions=reactions_ABC, output_order=['A', 'B', 'C'], verbose=False)
    result = solve_ivp(
        drl.calculate_step, t_span=[time[0], time[-1]], y0=[A0, 0, 0], method='Radau', t_eval=time,
        jac=drl.calculate_jac, rtol=rtol, atol=atol)
    MAPE_A = mean_absolute_percentage_error(y_pred=result.y[0], y_true=kinetic_A)
    MAPE_B = mean_absolute_percentage_error(y_pred=result.y[1], y_true=kinetic_B)
    MAPE_C = mean_absolute_percentage_error(y_pred=result.y[2], y_true=kinetic_C)
    time_prediction = perf_counter() - ti
    print(MAPE_A + MAPE_B + MAPE_C)
    print(f"calculated in {time_prediction:4f} seconds")

    assert time_prediction < 0.1
    assert MAPE_A < rtol*10
    assert MAPE_B < rtol*10
    assert MAPE_C < rtol*10

    # predict
    print('\nDRL Euler')
    ti = perf_counter()
    drl = DRL(rate_constants=rate_constants, reactions=reactions_ABC, verbose=False, output_order=['A', 'B', 'C'])
    pred = drl.predict_concentration_Euler(
        t_eval_pre=np.linspace(0, 1, 10),
        t_eval_post=time,
        initial_concentrations={},
        labeled_concentration={'A': 1},
        dilution_factor=1,
    )
    MAPE_A = mean_absolute_percentage_error(y_pred=pred['A'], y_true=kinetic_A)
    MAPE_B = mean_absolute_percentage_error(y_pred=pred['B'], y_true=kinetic_B)
    MAPE_C = mean_absolute_percentage_error(y_pred=pred['C'], y_true=kinetic_C)
    time_prediction = perf_counter() - ti
    print(MAPE_A + MAPE_B + MAPE_C)
    print(f"calculated in {time_prediction:4f} seconds")

    assert time_prediction < 0.6
    assert MAPE_A < 0.01
    assert MAPE_B < 0.01
    assert MAPE_C < 0.01
