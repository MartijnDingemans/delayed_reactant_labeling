from __future__ import annotations

import pytest
from pytest import approx, raises
import numpy as np
from delayed_reactant_labeling.predict import DRL
from delayed_reactant_labeling.optimize import RateConstantOptimizerTemplate
import pandas as pd
from scipy.optimize import Bounds
from icecream import ic

reactions = [
    ('k1', ['A', 'cat'], ['B'],),
    ('k-1', ['B'], ['A', 'cat'],),
    ('k2', ['B'], ['C', 'cat']),

    # labeled
    ('k1', ['A-d10', 'cat'], ['B-d10'],),
    ('k-1', ['B-d10'], ['A-d10', 'cat'],),
    ('k2', ['B-d10'], ['C-d10', 'cat'])
]

# look at as simple of a system as possible.
concentration_initial = {'A': 1, 'cat': 1 / 5}
concentration_labeled = {'A-d10': 1}
dilution_factor = 1
time_pre = np.linspace(0, 10, 50)
time_post = np.linspace(10, 90, 8 * 50)

# create a "real" prediction.
rate_constants_real = {'k1': 0.3, 'k-1': 0.05, 'k2': 0.5}
drl_real = DRL(rate_constants=rate_constants_real, reactions=reactions)
real_data = drl_real.predict_concentration(
    t_eval_pre=time_pre,
    t_eval_post=time_post,
    dilution_factor=dilution_factor,
    initial_concentrations=concentration_initial,
    labeled_concentration=concentration_labeled)

# add noise
rng = np.random.default_rng(42)
fake_data = []
for col in real_data.columns[:-1]:  # the last column contains a time array, so skip that one.
    noise_dynamic = rng.normal(loc=1, scale=0.05, size=real_data[col].shape)  # fraction error
    noise_static = rng.normal(loc=0.015, scale=0.0075, size=real_data[col].shape)
    fake_col = real_data[col] * noise_dynamic + noise_static
    fake_col[fake_col < 1e-10] = 1e-10  # no negative intensity
    fake_data.append(fake_col)
fake_data.append(real_data['time'])
fake_data = pd.DataFrame(fake_data, index=real_data.columns).T


class RateConstantOptimizer(RateConstantOptimizerTemplate):
    @staticmethod
    def create_prediction(x: np.ndarray, x_description: list[str]) -> pd.DataFrame:
        rate_constants = pd.Series(x, x_description)
        # The rate constants can easily be manipulated here. For example,
        # rate_constants["k1"] = 0.42 would fixate that value.
        # Because Series are mutable, the changed version will be stored in the logs!

        drl = DRL(reactions=reactions, rate_constants=rate_constants)
        pred_labeled = drl.predict_concentration(
            t_eval_pre=time_pre,
            t_eval_post=time_post,
            initial_concentrations=concentration_initial,
            labeled_concentration=concentration_labeled,
            dilution_factor=dilution_factor,
            rtol=1e-8,
            atol=1e-8, )

        # The prediction can be altered here before its analyzed.
        return pred_labeled

    @staticmethod
    def calculate_curves(data: pd.DataFrame) -> dict[str, np.ndarray]:
        curves = {}
        for chemical in ['A', 'B', 'C']:
            chemical_sum = data[[chemical, f'{chemical}-d10']].sum(axis=1)
            curves[f'ratio_{chemical}'] = (data[chemical] / chemical_sum).to_numpy()
        return curves


def metric(y_true, y_pred):
    return np.average( np.abs(y_true - y_pred))


x_description = ['k1', 'k-1', 'k2']
bounds = Bounds(np.array([1e-9, 1e-9, 1e-9]), np.array([100, 100, 100]))  #lower bound, upper bound

RCO = RateConstantOptimizer(experimental=fake_data, metric=metric)


def test_RCO():
    RCO.optimize(
        x0=np.array([1, 1, 1]),
        x_description=x_description,
        x_bounds=bounds,
        path='./optimization/',
        show_pbar=False,
        maxiter=20,                 # otherwise 262 iterations are performed, takes roughly 20 seconds
        _overwrite_log=True)

    model = RCO.load_optimized_model('./optimization/', )

    assert np.allclose(model.optimal_x, [0.488754, 1.439228, 0.915710])  # not a good fit, but show the same pattern!
    # complete optimization should result to:
    # k1     2.112738e-01
    # k-1    1.000000e-09
    # k2     6.425392e-01


def test_weights():
    RCO_weighted = RateConstantOptimizer(experimental=fake_data, metric=metric, raw_weights={'ratio_A': 0.69})
    pred = RCO_weighted.create_prediction(np.array([1, 1, 1]), x_description=x_description)
    errors = RCO_weighted.calculate_errors(pred)
    weighted_errors = RCO_weighted.weigh_errors(errors)

    assert weighted_errors['ratio_A'] == approx(errors['ratio_A'] * 0.69)
    assert weighted_errors['ratio_B'] == errors['ratio_B']


def test_throws_error_upon_nan_error():
    fake_data_faulty = fake_data.copy()
    # signal for both A and A-labeled drops to zero, corresponding curve should become nan
    fake_data_faulty.iloc[3:5, 0] = 0
    fake_data_faulty.iloc[2:4, 1] = 0

    with pytest.warns():
        _RCO = RateConstantOptimizer(experimental=fake_data_faulty, metric=metric)

    pred = _RCO.create_prediction(np.array([1, 1, 1]), x_description=x_description)

    with pytest.raises(ValueError):
        _RCO.calculate_errors(pred)

