import unittest
from time import perf_counter

import numpy as np
from scipy.integrate import solve_ivp
from sklearn.metrics import mean_absolute_percentage_error

from delayed_reactant_labeling.predict import DRL


NUMBA_CAPTURED_ERRORS='new_style'


class MyTestCase(unittest.TestCase):
    def test_jac_simple(self):
        reactions_ABC = [
            ("k1", ["A"], ["B"]),
            ("k2", ["B"], ["C"]),
        ]
        rate_constants_ABC = {
            "k1": .2,
            "k2": .5,
        }

        k1 = rate_constants_ABC['k1']
        k2 = rate_constants_ABC['k2']

        drl = DRL(reactions=reactions_ABC, rate_constants=rate_constants_ABC)
        J = drl.calculate_jac(None, np.array([1, 0, 0]))

        J_expected_100 = np.array([
            [-k1, 0, 0],
            [k1, -k2, 0],
            [0, k2, 0],
        ])
        self.assertTrue(np.allclose(J, J_expected_100))

    def test_jac_moderate(self):
        reactions = [
            ('k1', ['cat', 'A'], ['B'],),
            ('k2', ['B'], ['C', 'cat'],)
        ]
        rate_constants = {
            "k1": .2,
            "k2": .5,
        }

        k1 = rate_constants['k1']
        k2 = rate_constants['k2']
        cat0 = 2
        A0 = 2

        drl = DRL(reactions=reactions, rate_constants=rate_constants, output_order=['cat', 'A', 'B', 'C'])
        J = drl.calculate_jac(None, np.array([cat0, A0, 0, 0]))

        J_expected = np.array([
            [-k1 * A0, -k1 * cat0, k2, 0],
            [-k1 * A0, -k1 * cat0, 0, 0],
            [k1 * A0, k1 * cat0, -k2, 0],
            [0, 0, k2, 0],
        ])
        self.assertTrue(np.allclose(J, J_expected))

    def test_drl(self):
        atol = 1e-10
        rtol = atol

        reactions_ABC = [
            ("k1", ["A"], ["B"]),
            ("k2", ["B"], ["C"]),
        ]
        rate_constants_ABC = {
            "k1": .2,
            "k2": .5,
        }

        k1 = rate_constants_ABC['k1']
        k2 = rate_constants_ABC['k2']
        A0 = 1
        time = np.linspace(0, 20, 1000)

        # algebraic solution
        kinetic_A = A0 * np.exp(-k1 * time)
        kinetic_B = k1 / (k2 - k1) * A0 * (np.exp(-k1 * time) - np.exp(-k2 * time))
        kinetic_C = A0 * (1 - np.exp(-k1 * time) - k1 / (k2 - k1) * (np.exp(-k1 * time) - np.exp(-k2 * time)))

        # make sure everything has compiled
        drl = DRL(
            rate_constants=rate_constants_ABC, reactions=reactions_ABC, output_order=['A', 'B', 'C'], verbose=False)
        _ = solve_ivp(
            drl.calculate_step, t_span=[time[0], time[-1]], y0=[A0, 0, 0], method='Radau', t_eval=time, jac=drl.calculate_jac,
            rtol=rtol, atol=atol)
        _ = solve_ivp(
            drl.calculate_step, t_span=[time[0], time[-1]], y0=[A0, 0, 0], method='Radau', t_eval=time, jac=None,
            rtol=rtol, atol=atol)

        # predict new
        ti = perf_counter()
        print('DRL new, no jac')
        drl = DRL(
            rate_constants=rate_constants_ABC, reactions=reactions_ABC, output_order=['A', 'B', 'C'], verbose=False)
        result = solve_ivp(
            drl.calculate_step, t_span=[time[0], time[-1]], y0=[A0, 0, 0], method='Radau', t_eval=time,   jac=None,
            rtol=rtol, atol=atol)
        MAPE_A = mean_absolute_percentage_error(y_pred=result.y[0], y_true=kinetic_A)
        MAPE_B = mean_absolute_percentage_error(y_pred=result.y[1], y_true=kinetic_B)
        MAPE_C = mean_absolute_percentage_error(y_pred=result.y[2], y_true=kinetic_C)
        print(MAPE_A + MAPE_B + MAPE_C)
        print(f"calculated in {perf_counter() - ti:4f} seconds")

        print('DRL new, automatic jac')
        ti = perf_counter()
        drl = DRL(
            rate_constants=rate_constants_ABC, reactions=reactions_ABC, output_order=['A', 'B', 'C'], verbose=False)
        result = solve_ivp(
            drl.calculate_step, t_span=[time[0], time[-1]], y0=[A0, 0, 0], method='Radau', t_eval=time,
            jac=drl.calculate_jac, rtol=rtol, atol=atol)
        MAPE_A = mean_absolute_percentage_error(y_pred=result.y[0], y_true=kinetic_A)
        MAPE_B = mean_absolute_percentage_error(y_pred=result.y[1], y_true=kinetic_B)
        MAPE_C = mean_absolute_percentage_error(y_pred=result.y[2], y_true=kinetic_C)
        print(MAPE_A + MAPE_B + MAPE_C)
        print(f"calculated in {perf_counter() - ti:4f} seconds")

        # check error with these statements
        self.assertLessEqual(MAPE_A, rtol*10)
        self.assertLessEqual(MAPE_B, rtol*10)
        self.assertLessEqual(MAPE_C, rtol*10)

        # TODO checkout at home why J[2, 1] = 0 performs better

        # predict
        print('DRL Euler')
        ti = perf_counter()
        drl = DRL(rate_constants=rate_constants_ABC, reactions=reactions_ABC, verbose=False, output_order=['A', 'B', 'C'])
        pred, _ = drl.predict_concentration_Euler(
            t_eval_pre=time,
            t_eval_post=np.linspace(0, 1, 10),
            initial_concentrations={'A': 1},
            labeled_concentration={},
            dilution_factor=1,
        )
        MAPE_A = mean_absolute_percentage_error(y_pred=pred['A'], y_true=kinetic_A)
        MAPE_B = mean_absolute_percentage_error(y_pred=pred['B'], y_true=kinetic_B)
        MAPE_C = mean_absolute_percentage_error(y_pred=pred['C'], y_true=kinetic_C)
        print(MAPE_A + MAPE_B + MAPE_C)
        print(f"calculated in {perf_counter() - ti:4f} seconds")


if __name__ == '__main__':
    unittest.main()
