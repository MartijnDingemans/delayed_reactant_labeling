from __future__ import annotations

import os
import warnings

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

import numpy as np
import pandas as pd

from scipy.optimize import Bounds
from delayed_reactant_labeling.predict import DRL
from delayed_reactant_labeling.optimize import RateConstantOptimizerTemplate

USE_NOISE = True
USE_TIC = True
warnings.warn(f"TIC: {USE_TIC}, NOISE {USE_NOISE}")  # warnings.warn so its shows up in the log easily

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


class RateConstantOptimizer(RateConstantOptimizerTemplate):
    @staticmethod
    def create_prediction(x: np.ndarray | list[float], x_description: list[str]) -> pd.DataFrame:
        rate_constants = pd.Series(x, x_description)
        drl = DRL(reactions=reactions, rate_constants=rate_constants)
        pred_labeled = drl.predict_concentration(
            t_eval_pre=time_pre,
            t_eval_post=time_post,
            initial_concentrations=concentration_initial,
            labeled_concentration=concentration_labeled,
            dilution_factor=dilution_factor,
            rtol=1e-8,
            atol=1e-8, )
        return pred_labeled

    @staticmethod
    def calculate_curves(data: pd.DataFrame) -> dict[str, np.ndarray]:
        curves = {}
        for chemical in ['A', 'B', 'C']:
            chemical_sum = data[[chemical, f'{chemical}-d10']].sum(axis=1)
            curves[f'ratio_{chemical}'] = (data[chemical] / chemical_sum).to_numpy()

            if USE_TIC:
                curves[f'TIC_{chemical}'] = (data[chemical] / chemical_sum.mean())
                curves[f'TIC_{chemical}-d10'] = (data[f'{chemical}-d10'] / chemical_sum.mean())
        return curves

    def calculate_errors(self, prediction: pd.DataFrame) -> pd.Series:
        errors = super().calculate_errors(prediction)
        if USE_TIC:
            ratio = errors[errors.index.str.contains("ratio")].sum() / (errors[errors.index.str.contains("TIC")].sum()/2)
            if 0.05 > ratio > 20:
                warnings.warn(f"errors are disproportional: {errors}")
        return errors


def METRIC(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.average(np.abs(y_pred - y_true), axis=0)


def explore_boundary(path, k1, kr1, k2, noise_func):
    path = f'{path}/k1_{k1}_kr1_{kr1}_k2_{k2}/'
    os.mkdir(path)

    # "real" fake data
    rate_constants_real = {'k1': k1, 'k-1': kr1, 'k2': k2}
    drl_real = DRL(rate_constants=rate_constants_real, reactions=reactions)
    real_data = drl_real.predict_concentration(
        t_eval_pre=time_pre,
        t_eval_post=time_post,
        dilution_factor=dilution_factor,
        initial_concentrations=concentration_initial,
        labeled_concentration=concentration_labeled)

    # model the ionization efficiency
    real_data['cat'] = real_data['cat'] * 1
    real_data['A'] = real_data['A'] * 20
    real_data['A-d10'] = real_data['A-d10'] * 20
    real_data['B'] = real_data['B'] * 0.5
    real_data['B-d10'] = real_data['B-d10'] * 0.5
    real_data['C'] = real_data['C'] * 3
    real_data['C-d10'] = real_data['C-d10'] * 3

    fig, ax = plt.subplots()
    real_data.plot('time', ax=ax)

    fig.savefig(f'{path}/real_data.png', dpi=200)

    # add noise
    fake_data = []
    rng = np.random.default_rng(42)
    for col in real_data.columns[:-1]:  # last column contains time array
        fake_col = noise_func(rng, real_data[col].values)
        fake_col[fake_col < 1e-10] = 1e-10  # no negative intensity
        fake_data.append(fake_col)
        ax.scatter(real_data['time'], fake_col, label=col, marker='.')

    fake_data.append(real_data['time'])
    fake_data = pd.DataFrame(fake_data, index=real_data.columns).T
    fig.savefig(f'{path}/fake_data.png', dpi=200)

    ax.set_yscale("log")
    fig.savefig(f'{path}/fake_data_logscale.png', dpi=200)
    plt.close(fig)

    RCO = RateConstantOptimizer(raw_weights={}, experimental=fake_data, metric=METRIC)
    dimension_description = ['k1', 'k-1', 'k2']
    x_bounds = Bounds(np.array([1e-9, 0, 1e-9]), np.array([100, 100, 100]))  # lower, upper
    RCO.optimize_multiple(path=f'{path}/multiple_guess/', n_runs=1000, x_bounds=x_bounds,
                          x_description=dimension_description, n_jobs=-1,
                          metadata={"USE_NOISE": USE_NOISE, "USE_TIC": USE_TIC})


def noisify(rng: np.random.Generator, arr: np.ndarray) -> np.ndarray:
    if USE_NOISE:
        noise_dynamic = rng.normal(loc=1, scale=0.1, size=arr.shape)  # fraction error
        noise_static = rng.normal(loc=0.015, scale=0.0075, size=arr.shape)
        return arr * noise_dynamic + noise_static
    else:
        return arr


rate_values = [0.01, 0.1, 1, 5]
for _k1 in rate_values:
    for _kr1 in rate_values:
        for _k2 in rate_values:
            try:
                explore_boundary(f'./optimization/', _k1, _kr1, _k2, noise_func=noisify)
            except Exception as e:
                warnings.warn(f"{_k1}, {_kr1}, {_k2} yielded error: {e}")
