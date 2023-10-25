import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
import numpy as np
from delayed_reactant_labeling.predict import DRL, Experimental_Conditions, InvalidPredictionError
import unittest


class MyTestCase(unittest.TestCase):
    def test_drl(self):

        import matplotlib.pyplot as plt

        reaction = [
            ("k1", ["A"], ["B"]),
            ("k2", ["B"], ["C"]),
        ]

        constants = {
            "k1": .2,
            "k2": .5,
        }

        A0 = 1
        experimental_conditions = Experimental_Conditions(
            time=(np.linspace(0, 20, 2000), np.linspace(0, 10, 10)),
            initial_concentrations={'A': A0},
            dilution_factor=1,
            labeled_reactant={}
        )

        time = experimental_conditions.time[0]
        k1 = constants['k1']
        k2 = constants['k2']
        kinetic_A = A0 * np.exp(-k1 * time)
        kinetic_B = k1 / (k2 - k1) * A0 * (np.exp(-k1 * time) - np.exp(-k2 * time))
        kinetic_C = A0 * (1 - np.exp(-k1 * time) - k1 / (k2 - k1) * (np.exp(-k1 * time) - np.exp(-k2 * time)))

        drl = DRL(rate_constants=constants, reactions=reaction, verbose=False)
        pred, _ = drl.predict_concentration(experimental_conditions=experimental_conditions)

        MAPE_A = mean_absolute_percentage_error(y_pred=pred['A'], y_true=kinetic_A)
        MAPE_B = mean_absolute_percentage_error(y_pred=pred['B'], y_true=kinetic_B)
        MAPE_C = mean_absolute_percentage_error(y_pred=pred['C'], y_true=kinetic_C)

        self.assertLessEqual(MAPE_A+MAPE_B+MAPE_C, 0.01)


if __name__ == '__main__':
    unittest.main()
