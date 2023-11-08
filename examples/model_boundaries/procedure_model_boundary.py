from __future__ import annotations

import pathlib
import shutil
import warnings
import zipfile

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')  # when committing to the server, this is essential!

import numpy as np
import pandas as pd

from scipy.optimize import Bounds
from delayed_reactant_labeling.predict import DRL
from delayed_reactant_labeling.optimize import RateConstantOptimizerTemplate
from delayed_reactant_labeling.visualize import VisualizeMultipleSolutions

def analyze_model_boundary(use_noise, use_tic, optimize=False):
    warnings.warn(f"TIC: {use_tic}, NOISE {use_noise}")  # warnings.warn so its shows up in the log easily

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

                if use_tic:
                    curves[f'TIC_{chemical}'] = (data[chemical] / chemical_sum.mean())
                    curves[f'TIC_{chemical}-d10'] = (data[f'{chemical}-d10'] / chemical_sum.mean())
            return curves

        def calculate_errors(self, prediction: pd.DataFrame) -> pd.Series:
            errors = super().calculate_errors(prediction)
            if use_tic:
                ratio = errors[errors.index.str.contains("ratio")].sum() / (
                            errors[errors.index.str.contains("TIC")].sum() / 2)
                if 0.05 > ratio > 20:
                    warnings.warn(f"errors are disproportional: {errors}")
            return errors

    def METRIC(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.average(np.abs(y_pred - y_true), axis=0)

    def noisify(rng: np.random.Generator, arr: np.ndarray) -> np.ndarray:
        if use_noise:
            noise_dynamic = rng.normal(loc=1, scale=0.1, size=arr.shape)  # fraction error
            noise_static = rng.normal(loc=0.015, scale=0.0075, size=arr.shape)
            return arr * noise_dynamic + noise_static
        else:
            return arr

    def to_zip(directory: pathlib.Path):
        with zipfile.ZipFile(f'{directory}.zip', mode='x', compression=zipfile.ZIP_DEFLATED) as archive:
            for file_path in directory.rglob("*"):
                archive.write(file_path, arcname=file_path.relative_to(directory))
        shutil.rmtree(directory)

    def explore_boundary(path, k1, kr1, k2, noise_func):
        path = pathlib.Path(f'{path}/k1_{k1}_kr1_{kr1}_k2_{k2}/')
        path.mkdir(exist_ok=True, parents=True)

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

        if optimize:
            RCO.optimize_multiple(path=f'{path}/multiple_guess/', n_runs=1000, x_bounds=x_bounds,
                                  x_description=dimension_description, n_jobs=-1,
                                  metadata={"USE_NOISE": use_noise, "USE_TIC": use_tic})
            to_zip(path)  # individual files take up at least 1 Mb at the server, -> zipping reduces size load drastically.

        # analysis of the data
        with zipfile.ZipFile(f'optimization/{path}.zip', mode="r") as archive:
            archive.extractall(path=f'optimization/{path}')
        VMS = VisualizeMultipleSolutions(f'optimization/{path}/multiple_guess/')
        shutil.rmtree(f'optimization/{path}')

        # using the "real" values
        pred = RCO.create_prediction(x=[k1, kr1, k2], x_description=dimension_description)
        errors = RCO.calculate_errors(pred)
        total_error_real = RCO.calculate_total_error(errors)

        # using best found values
        sorted_index = VMS.complete_found_error.argsort()
        best_run = sorted_index[0]
        best_X = pd.Series(VMS.complete_optimal_X[best_run], index=VMS.x_description)

        out = pd.Series({
            "k1": k1,
            "kr1": kr1,
            "k2": k2,
            "A_td": fake_data.loc[40:60, "A"].mean(),
            "B_td": fake_data.loc[40:60, "B"].mean(),
            "C_td": fake_data.loc[40:60, "C"].mean(),
            "real_error": total_error_real,
            "best_error": VMS.complete_found_error[best_run],
            "best_k1": best_X["k1"],
            "best_kr1": best_X["k-1"],
            "best_k2": best_X["k2"],
            "k1_ratio": best_X["k1"] / k1,
            "kr1_ratio": best_X["k-1"] / kr1,
            "k2_ratio": best_X["k2"] / k2,
        })
        return out

    data = []
    rate_values = [0.01, 0.1, 1, 5]
    for _k1 in rate_values:
        for _kr1 in rate_values:
            for _k2 in rate_values:
                try:
                    data.append(explore_boundary(f'./n_{use_noise}_t_{use_tic}/', _k1, _kr1, _k2, noise_func=noisify))
                except Exception as e:
                    warnings.warn(f"k1_{_k1}_kr1_{_kr1}_k2_{_k2} yielded error: {e}")
    df = pd.DataFrame(data)
    df.to_excel(f"bounds_n_{use_noise}_t_{use_tic}.xlsx")


if __name__ == "__main__":
    analyze_model_boundary(use_noise=True, use_tic=False, optimize=False)
