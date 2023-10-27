import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
from scipy.integrate import solve_ivp
from delayed_reactant_labeling.predict import DRL, Experimental_Conditions, InvalidPredictionError
from delayed_reactant_labeling.predict_new import DRL as DRL_new
from time import perf_counter
import unittest


class MyTestCase(unittest.TestCase):
    def test_jac(self):
        return
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

        drl = DRL_new(reactions=reactions_ABC, rate_constants=rate_constants_ABC)

        J = drl.calculate_jac(None, np.array([1, 0, 0]))
        J_expected_100 = np.array([
            [-k1, 0, 0],
            [k1, -k2, 0],
            [0, 0, 0],
        ])
        self.assertTrue(np.allclose(J, np.array(J_expected_100)))

        J = drl.calculate_jac(None, [1, 1, 0])
        J_expected_110 = np.array([
            [-k1, 0, 0],
            [k1, -k2, 0],
            [0, k2, 0],
        ])
        self.assertTrue(np.allclose(J, np.array(J_expected_110)))

        J = drl.calculate_jac(None, [1, 1, 1])
        J_expected_111 = np.array([
            [-k1, 0, 0],
            [k1, -k2, 0],
            [0, k2, 0],
        ])
        self.assertTrue(np.allclose(J, J_expected_111))

    def test_drl(self):
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

        # predict new
        ti = perf_counter()
        print('DRL new, no jac')
        drl = DRL_new(rate_constants=rate_constants_ABC, reactions=reactions_ABC, output_order=['A', 'B', 'C'], verbose=False)
        result = solve_ivp(drl.calculate_step, t_span=[time[0], time[-1]], y0=[A0, 0, 0], method='Radau', t_eval=time, jac=None)
        MAPE_A = mean_absolute_percentage_error(y_pred=result.y[0], y_true=kinetic_A)
        MAPE_B = mean_absolute_percentage_error(y_pred=result.y[1], y_true=kinetic_B)
        MAPE_C = mean_absolute_percentage_error(y_pred=result.y[2], y_true=kinetic_C)
        print(MAPE_A + MAPE_B + MAPE_C)
        print(f"calculated in {perf_counter()-ti:4f} seconds")

        # TODO checkout at home why J[2, 1] = 0 performs better
        print('DRL new, automatic jac')
        ti = perf_counter()
        drl = DRL_new(rate_constants=rate_constants_ABC, reactions=reactions_ABC, output_order=['A', 'B', 'C'], verbose=False)
        result = solve_ivp(drl.calculate_step, t_span=[time[0], time[-1]], y0=[A0, 0, 0], method='Radau', t_eval=time, jac=drl.calculate_jac)
        MAPE_A = mean_absolute_percentage_error(y_pred=result.y[0], y_true=kinetic_A)
        MAPE_B = mean_absolute_percentage_error(y_pred=result.y[1], y_true=kinetic_B)
        MAPE_C = mean_absolute_percentage_error(y_pred=result.y[2], y_true=kinetic_C)
        print(MAPE_A + MAPE_B + MAPE_C)
        print(f"calculated in {perf_counter()-ti:4f} seconds")

        # predict
        experimental_conditions = Experimental_Conditions(
            time=(time, np.linspace(0, 10, 10)),
            initial_concentrations={'A': A0},
            dilution_factor=1,
            labeled_reactant={}
        )
        drl = DRL(rate_constants=rate_constants_ABC, reactions=reactions_ABC, verbose=False)
        pred, _ = drl.predict_concentration(experimental_conditions=experimental_conditions)

        MAPE_A = mean_absolute_percentage_error(y_pred=pred['A'], y_true=kinetic_A)
        MAPE_B = mean_absolute_percentage_error(y_pred=pred['B'], y_true=kinetic_B)
        MAPE_C = mean_absolute_percentage_error(y_pred=pred['C'], y_true=kinetic_C)
        print(MAPE_A + MAPE_B + MAPE_C)


if __name__ == '__main__':
    unittest.main()
