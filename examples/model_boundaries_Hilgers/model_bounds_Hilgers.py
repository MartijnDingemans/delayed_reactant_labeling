from __future__ import annotations

import os
import pathlib
import warnings
from copy import deepcopy

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')
import numpy as np
import pandas as pd
from scipy.optimize import minimize, Bounds

from delayed_reactant_labeling.optimize import RateConstantOptimizerTemplate, OptimizedMultipleModels
from delayed_reactant_labeling.predict import DRL

NOISE_LEVEL = 0.02
STATIC_NOISE = True
RATE_CONSTANTS_ROELANT = pd.Series({
    'k1_D': 1.5,
    'k1_E': 0.25,
    'k1_F': 0.01,
    'k2_D': 0.43,
    'k2_E': 0.638,
    'k2_F': 0.567,
    'k3_D': 0.23,
    'k3_E': 0.35,
    'k3_F': 0.3,
    'k4_D': 8,
    'k4_E': 0.05,
    'k4_F': 0.03,
    'k-1_D': 0,
    'k-1_E': 0,
    'k-1_F': 0,
    'k-2_D': 0.025,
    'k-2_E': 0.035,
    'k-2_F': 0.03,
    'k-3_D': 0,
    'k-3_E': 0,
    'k-3_F': 0,
    'k-4_D': 0,
    'k-4_E': 0,
    'k-4_F': 0,
})
LABEL = "'"
ISOMERS = ['D', 'E', 'F']
TIME_PRE = np.linspace(0, 10, 1000)
TIME_POST = np.linspace(10, 40, 3000)
CONC_INITIAL = {'cat': 0.005 * 40 / 1200,  # concentration in M
                '2': 0.005 * 800 / 1200}
CONC_LABELED = {f'2{LABEL}': 0.005 * 800 / 2000}
DILUTION_FACTOR = 1200 / 2000
WEIGHTS = {
    "label_": 1,
    "isomer_": 0.5,
    "TIC": 0.2,
    "iso_F": 0.25,
}


def create_reactions():
    REACTIONS_ONEWAY = []
    for label in ['', LABEL]:
        REACTIONS_ONEWAY.extend([
            ('k1_D', ['cat', f'2{label}', ], [f'3D{label}', ]),
            ('k1_E', ['cat', f'2{label}', ], [f'3E{label}', ]),
            ('k1_F', ['cat', f'2{label}', ], [f'3F{label}', ]),

            ('k2_D', [f'3D{label}', ], [f'4D{label}', ]),
            ('k2_E', [f'3E{label}', ], [f'4E{label}', ]),
            ('k2_F', [f'3F{label}', ], [f'4F{label}', ]),

            ('k3_D', [f'4D{label}', ], [f'5D{label}', ]),
            ('k3_E', [f'4E{label}', ], [f'5E{label}', ]),
            ('k3_F', [f'4F{label}', ], [f'5F{label}', ]),

            ('k4_D', [f'5D{label}', ], [f'6D{label}', 'cat', ]),
            ('k4_E', [f'5E{label}', ], [f'6E{label}', 'cat', ]),
            ('k4_F', [f'5F{label}', ], [f'6F{label}', 'cat', ]),
        ])

    _reactions = deepcopy(REACTIONS_ONEWAY)
    for k, reactants, products in REACTIONS_ONEWAY:
        _reactions.append(('k-' + k[1:], products, reactants))
    return _reactions


reactions = create_reactions()
rate_constant_names = sorted(set([k for k, _, _ in reactions]))
WEIGHT_TIME = 1 - 0.9 * np.linspace(0, 1, TIME_POST.shape[0])  # decrease weight with time
WEIGHT_TIME = WEIGHT_TIME / sum(WEIGHT_TIME)  # normalize


def METRIC(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.average(np.abs(y_pred - y_true), weights=WEIGHT_TIME, axis=0)


def optimize(compounds: list[str], calculate=True, noise_level=NOISE_LEVEL, static_noise=STATIC_NOISE):
    path = f'./optimization/noise_{noise_level}_compounds_{"_".join(compounds).replace("/", "and")}'
    if static_noise:
        path = f'{path}_STATIC'
    path = pathlib.Path(path)

    if calculate:
        path.mkdir(exist_ok=False, parents=True)

    compounds = [compound.replace('and', '/') for compound in compounds]
    combine_4_5 = '4/5' in compounds

    def plot_data(_data):
        fig, axs = plt.subplots(len(compounds), sharex='all', figsize=(9, 10))
        for ax, compound in zip(axs, compounds):
            if compound == 'cat':
                ax.plot(TIME_POST, _data['cat'], label='cat')
            else:
                for label in ['', LABEL]:
                    if compound == "2":
                        ax.plot(TIME_POST, _data[f'2{label}'], label=f'2{label}')
                    else:
                        for isomer in ISOMERS:
                            ax.plot(TIME_POST, _data[f'{compound}{isomer}{label}'],
                                    label=f'{compound}{isomer}{label}')

        for ax in axs:
            ax.legend(ncol=6)
            # ax.set_yscale('log')
        return fig, axs

    class RateConstantOptimizer(RateConstantOptimizerTemplate):
        @staticmethod
        def create_prediction(x: np.ndarray, x_description: list[str]) -> pd.DataFrame:
            # separate out the ionization factor from the other parameters which are being optimized.
            rate_constants = pd.Series(x[:len(rate_constant_names)], index=x_description[:len(rate_constant_names)])
            ionization_factor = x[-1]

            drl = DRL(reactions=reactions,
                      rate_constants=rate_constants,
                      verbose=False)

            prediction_labeled = drl.predict_concentration(
                t_eval_pre=TIME_PRE,
                t_eval_post=TIME_POST,
                initial_concentrations=CONC_INITIAL,
                labeled_concentration=CONC_LABELED,
                dilution_factor=DILUTION_FACTOR)

            if not combine_4_5:
                return prediction_labeled

            # SYSTEM-SPECIFIC ENAMINE IONIZATION CORRECTION -> only a prediction of 4 and 5 together can be made!
            # this because the unstable enamine will ionize to the iminium ion upon injection in the mass spectrometer.
            for isomer in ISOMERS:
                for label in ["", "'"]:
                    prediction_labeled.loc[:, f"4/5{isomer}{label}"] = \
                        prediction_labeled.loc[:, f"5{isomer}{label}"] \
                        + ionization_factor * prediction_labeled.loc[:, f"4{isomer}{label}"]
            return prediction_labeled

        @staticmethod
        def calculate_curves(data: pd.DataFrame) -> dict[str, np.ndarray]:
            curves = {}
            for compound in compounds:
                if compound == 'cat':
                    curves['TIC_int_cat'] = (data['cat'] / data['cat'].iloc[-100:].mean()).to_numpy()
                elif compound == '2':
                    sum_chemical = data[['2', f'2{LABEL}']].sum(axis=1)
                    curves['label_int_2'] = (data['2'] / sum_chemical).to_numpy()
                    curves['TIC_int_2'] = (data['2'] / sum_chemical.iloc[-100:].mean()).to_numpy()
                    curves[f'TIC_int_2{LABEL}'] = (data[f'2{LABEL}'] / sum_chemical.iloc[-100:].mean()).to_numpy()
                else:  # 3, 4, 5, 6, or 4/5
                    # sum does not have to be recalculated between the isomer runs
                    sum_all_isomers = data[[f'{compound}{isomer}' for isomer in ISOMERS]].sum(axis=1)
                    for isomer in ISOMERS:
                        # allows for easy modification of weight. str.contains('int_1') is much more specific than just '1'
                        chemical_iso_split = f"int_{compound}_iso_{isomer}"
                        chemical = f"{compound}{isomer}"  # 3D, 3E, 3F, 4/5D, 4/5E, 3/5F
                        sum_chemical = data[[chemical, f"{chemical}'"]].sum(axis=1)
                        curves[f"label_{chemical_iso_split}"] = (  # 3D / (3D+3D')
                                data[chemical] / sum_chemical).to_numpy()
                        curves[f"isomer_{chemical_iso_split}"] = (  # 3D / (3D+3E+3F)
                                data[chemical] / sum_all_isomers).to_numpy()
                        curves[f"TIC_{chemical_iso_split}"] = (  # normalized TIC curve
                                data[chemical] / sum_chemical.iloc[-100:].mean()).to_numpy()
                        curves[f"TIC_{chemical_iso_split}'"] = (  # normalized TIC curve
                                data[f"{chemical}'"] / sum_chemical.iloc[-100:].mean()).to_numpy()
            return curves

        def weigh_errors(self, errors: pd.Series) -> pd.Series:
            weighed_errors = super().weigh_errors(errors)

            # perform the usual behavior of this function, but also perform an additional check with regards to the output!
            TIC_sum = weighed_errors[weighed_errors.index.str.contains("TIC-")].sum()
            label_sum = weighed_errors[weighed_errors.index.str.contains("label_")].sum()
            isomer_sum = weighed_errors[weighed_errors.index.str.contains("isomer_")].sum()
            total = TIC_sum + label_sum + isomer_sum
            ratios = pd.Series([TIC_sum / total, label_sum / total, isomer_sum / total],
                               index=['TIC', 'label', 'total'])
            if any(ratios < 0.001) or any(ratios > 0.999):
                warnings.warn(
                    f'One of the error metrics is either way smaller, or way larger than the others\n{ratios}')

            return weighed_errors

    def create_data():
        if not path.exists():
            path.mkdir(parents=True)

        if combine_4_5:
            x = RATE_CONSTANTS_ROELANT.copy()
            x['ion'] = 0.025
        else:
            x = RATE_CONSTANTS_ROELANT.copy()
        real_data = RateConstantOptimizer.create_prediction(
            x=x.values, x_description=list(x.keys()))
        fig, axs = plot_data(real_data)
        fig.suptitle('raw concentrations')
        fig.tight_layout()
        fig.savefig(f'{path}/real_fake_data.png')

        compounds_ionization_factor = {'cat': 0.5, '2': 0.001, '3': 69, '4': 0.2, '5': 42, '6': 0.3, '4/5': 42}
        fake_data = []
        fake_data_columns = []
        rng = np.random.default_rng(42)

        def add_chemical(chemical, ion_factor, ):
            fake_data_columns.append(chemical)
            noise_dynamic = noise_level * rng.normal(loc=0, scale=1, size=real_data[chemical].size) * real_data[
                chemical]
            fake_col = ion_factor * (real_data[chemical] + noise_dynamic)
            if static_noise:
                fake_col += rng.normal(loc=0, scale=0.000001 / 4, size=real_data[chemical].size)
                fake_col[fake_col <= 1e-15] = 1e-15  # no negative numbers

            fake_data.append(fake_col)

        for compound in compounds:
            if compound == 'cat':
                add_chemical(compound, compounds_ionization_factor[compound])
            else:
                for label in ['', LABEL]:
                    if compound == "2":
                        add_chemical(f'{compound}{label}', compounds_ionization_factor[compound])
                    else:
                        for isomer in ISOMERS:
                            add_chemical(f'{compound}{isomer}{label}', compounds_ionization_factor[compound])

        # normalize w.r.t. TIC
        fake_data = pd.DataFrame(fake_data, index=fake_data_columns).T
        fake_data['time (min)'] = real_data['time']
        fig, axs = plot_data(fake_data)
        fig.suptitle('concentrations -> ionization factors')
        fig.tight_layout()
        fig.savefig(f'{path}/fake_data.png')
        return fake_data

    def create_constraints():
        dimension_descriptions = list(rate_constant_names)
        if combine_4_5:
            dimension_descriptions.append('ion')

        constraints = pd.DataFrame(np.full((len(dimension_descriptions), 3), np.nan),
                                   columns=["vertex", "lower", "upper"],
                                   index=dimension_descriptions)

        index_reverse_reaction = constraints.index.str.contains("k-")
        constraints.iloc[np.nonzero(~index_reverse_reaction)] = [1, 1e-9, 1e2]  # forwards; vertex, lower, upper
        constraints.iloc[np.nonzero(index_reverse_reaction)] = [0.5, 0, 1e2]  # backwards

        # special case
        if combine_4_5:
            constraints.iloc[np.nonzero(constraints.index.str.contains("ion"))] = [0.01, 1e-6, 1]

        # Steady state
        STEADY_STATE_CHEMICALS = ['3D', '3E', '3F']
        EQUILIBRIUM_LAST_N = 500
        steady_state_constraints = []
        for chemical in STEADY_STATE_CHEMICALS:
            # normalize for each steady state such that chemical + chemical' = 1 at equilibrium
            y_true_curve = experimental[f'{chemical}{LABEL}'] / experimental.loc[-EQUILIBRIUM_LAST_N:,
                                                                [chemical, f'{chemical}{LABEL}']].sum(axis=1).mean()

            def f(k):
                return y_true_curve.iloc[-EQUILIBRIUM_LAST_N:].mean() * (
                        1 - np.exp(-k * (TIME_POST - TIME_POST[0])))

            def MAE_f(x):
                return METRIC(y_true=y_true_curve, y_pred=f(x))

            result = minimize(MAE_f, x0=np.array([1]))

            # analyze sensitivity to deviations
            rates = np.linspace(0, 5 * result.x[0], num=500)
            errors = np.array([MAE_f(x) for x in rates])
            bounds_10pc = np.where(errors < 1.1 * result.fun)[0][[0, -1]]
            steady_state_constraints.append([result.x[0], rates[bounds_10pc[0]], rates[bounds_10pc[1]]])

        constraints.iloc[np.nonzero(constraints.index.str.contains("k2_D"))] = steady_state_constraints[0]
        constraints.iloc[np.nonzero(constraints.index.str.contains("k2_E"))] = steady_state_constraints[1]
        constraints.iloc[np.nonzero(constraints.index.str.contains("k2_F"))] = steady_state_constraints[2]

        # either chemically or experimentally determined to be zero
        constraints.iloc[np.nonzero(constraints.index.str.contains("k-1"))] = [0, 0, 0]
        constraints.iloc[np.nonzero(constraints.index.str.contains("k-3"))] = [0, 0, 0]
        constraints.iloc[np.nonzero(constraints.index.str.contains("k-4"))] = [0, 0, 0]
        return (
            constraints["vertex"].to_numpy(),
            Bounds(constraints['lower'].to_numpy(), constraints['upper'].to_numpy()),
            dimension_descriptions,
        )

    experimental = create_data()
    RCO = RateConstantOptimizer(experimental=experimental, metric=METRIC, raw_weights=WEIGHTS)
    vertex, bounds, x_description = create_constraints()

    if calculate:
        RCO.optimize_multiple(
            path=path,
            x_description=x_description,
            metadata={'compounds': compounds, 'noise_level': noise_level},
            n_runs=100,
            n_jobs=-1,
            x_bounds=bounds,
            maxiter=200000,
        )
    else:
        # "real" x / x description, from Hilgers et al.
        x = RATE_CONSTANTS_ROELANT.to_list()
        x_description = list(RATE_CONSTANTS_ROELANT.keys())
        x = np.array(x + [0.025]) if combine_4_5 else np.array(x)
        x_description = x_description + ['ion'] if combine_4_5 else x_description

        # using the "real" values
        pred_real = RCO.create_prediction(x=x, x_description=x_description)
        errors_real = RCO.calculate_errors(pred_real)
        total_error_real = RCO.calculate_total_error(errors_real)

        # using best found values
        models = OptimizedMultipleModels(path)
        optimal_x = models.best.optimal_x

        out = pd.Series({
            "noise": noise_level,
            "cat": "cat" in compounds,
            "2": "2" in compounds,
            "3": "3" in compounds,
            "4": "4" in compounds,
            "5": "5" in compounds,
            "4/5": "4/5" in compounds,
            "6": "6" in compounds,
            "real_error": total_error_real,
            "best_error": models.best.optimal_error,
            "n_runs": models.all_optimal_error.shape[0],
        }, name=path.parts[-1])

        real_rates = pd.Series(x, x_description)
        for index in optimal_x.keys():
            if real_rates[index] == 0:  # disabled reaction
                continue
            out[f'_{index}/real'] = optimal_x[index] / real_rates[index]

        for index in optimal_x.keys():
            out[index] = optimal_x[index]

        pred = RCO.create_prediction(optimal_x.values, x_description=models.x_description)
        errors = RCO.calculate_errors(pred)
        for descr, error in errors.items():
            out[descr] = error

        for descr, error in errors_real.items():
            out[f'real_{descr}'] = error

        try:
            from delayed_reactant_labeling.visualize import VisualizeModel
            VM = VisualizeModel(
                image_path=path,
                models=models,
                rate_constant_optimizer=RCO,
                plot_title=path.parts[-1].replace('_', ' '), extensions='.png')
            VM.plot_error_all_runs()
            VM.plot_error_all_runs(20, file_name='plot_error_all_runs_zoom_in')
            VM.plot_x_all_runs(slice(10), file_name='plot_x_all_runs')
        except Exception as e:
            warnings.warn(f'Exception occured when plotting at {path}\n{e}')

        return out


if __name__ == '__main__':
    df = []
    for _path in os.listdir('./optimization/'):
        # path = f'./optimization/noise_{noise_level}_compounds_{"_".join(compounds).replace("/", "and")}/'
        items = _path.split('_')
        assert items[0] == 'noise'
        assert items[2] == 'compounds'

        if items[-1] == 'STATIC':
            _compounds = items[3:-1]
            static = True
        else:
            _compounds = items[3:]
            static = False

        try:
            df.append(optimize(compounds=_compounds, calculate=False, noise_level=float(items[1]), static_noise=static))
        except Exception as e:
            warnings.warn(f'Warning occured at {_path}:\n{e}')
        plt.close('all')

    df = pd.DataFrame(df)
    # TODO sort index
    df.to_excel('model_boundaries.xlsx')
