from copy import deepcopy

import pandas as pd
import polars as pl  # just much more efficient than pandas, although not as easy to use.
import numpy as np

from icecream import ic

from delayed_reactant_labeling.predict_new import DRL
from delayed_reactant_labeling.optimize import RateConstantOptimizerTemplate
from delayed_reactant_labeling.visualize import VisualizeSingleSolution, VisualizeMultipleSolutions

experimental_complete = pl.read_excel("experimental_data_Roelant.xlsx", engine='openpyxl')
index_labeled_compound = np.argmax(experimental_complete["time (min)"] > 10.15)

time_pre_addition = experimental_complete['time (min)'][:index_labeled_compound].to_numpy()

# only look at the part after the addition of labeled compound
experimental = experimental_complete[index_labeled_compound:, :]
time = experimental['time (min)'].to_numpy()

WEIGHT_TIME = 1 - 0.9 * np.linspace(0, 1, time.shape[0])  # decrease weight with time, first point 10 times as import as last point
WEIGHT_TIME = WEIGHT_TIME / sum(WEIGHT_TIME)  # normalize


def METRIC(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.average(np.abs(y_pred - y_true), weights=WEIGHT_TIME, axis=0)


REACTIONS_ONEWAY = [
    # unlabeled
    ("k1_D", ["cat", "2", ], ["3D", ]),
    ("k1_E", ["cat", "2", ], ["3E", ]),
    ("k1_F", ["cat", "2", ], ["3F", ]),

    ("k2_D", ["3D", ], ["4D", ]),
    ("k2_E", ["3E", ], ["4E", ]),
    ("k2_F", ["3F", ], ["4F", ]),

    ("k3_D", ["4D", ], ["5D", ]),
    ("k3_E", ["4E", ], ["5E", ]),
    ("k3_F", ["4F", ], ["5F", ]),

    ("k4_D", ["5D", ], ["6D", "cat", ]),
    ("k4_E", ["5E", ], ["6E", "cat", ]),
    ("k4_F", ["5F", ], ["6F", "cat", ]),

    # labeled
    ("k1_D", ["cat", "2'", ], ["3D'", ]),
    ("k1_E", ["cat", "2'", ], ["3E'", ]),
    ("k1_F", ["cat", "2'", ], ["3F'", ]),

    ("k2_D", ["3D'", ], ["4D'", ]),
    ("k2_E", ["3E'", ], ["4E'", ]),
    ("k2_F", ["3F'", ], ["4F'", ]),

    ("k3_D", ["4D'", ], ["5D'", ]),
    ("k3_E", ["4E'", ], ["5E'", ]),
    ("k3_F", ["4F'", ], ["5F'", ]),

    ("k4_D", ["5D'", ], ["6D'", "cat", ]),
    ("k4_E", ["5E'", ], ["6E'", "cat", ]),
    ("k4_F", ["5F'", ], ["6F'", "cat", ]),
]

# these groups allow us to loop over some code more easily
INTERMEDIATES = ["3", "4/5"]
ISOMERS = ["D", "E", "F"]


def create_reaction_equations():
    """Create the reverse reaction of each pre-defined reaction."""
    reactions_two_way_labeled = deepcopy(REACTIONS_ONEWAY)
    for k, reactants, products in REACTIONS_ONEWAY:
        reactions_two_way_labeled.append(("k-" + k[1:], products, reactants))

    return reactions_two_way_labeled


reaction_equations = create_reaction_equations()
rate_constant_names = sorted(set([k for k, _, _ in reaction_equations]))  # extract each unique k value

COMPOUND_RATIO = ("6D", ["6D", "6E"])  # chemical, compared to chemicals

# Weigh each error. The first element of each tuple is matches against the error string, and the selected errors are subsequently scaled.
WEIGHTS = {
    "label_": 1,  # all label ratios
    "isomer_": 0.5,  # all isomer ratios
    "TIC": 0.1,
    # all TIC ratios; this error is counted twice, once each for both the non-labeled and the labeled curves
    "iso_F": 0.25,  # iso_F selects only isomer F; these weights are multiplicative in case of multiple matches
}


class RateConstantOptimizer(RateConstantOptimizerTemplate):
    @staticmethod
    def create_prediction(x: np.ndarray, x_description: list[str]) -> tuple[pl.DataFrame, float]:
        # separate out the ionization factor from the other parameters which are being optimized.
        rate_constants = pd.Series(x[:len(rate_constant_names)], index=x_description[:len(rate_constant_names)])
        ionization_factor = x[-1]

        drl = DRL(reactions=reaction_equations,
                  rate_constants=rate_constants,
                  verbose=False)  # stores values in drl.reactions which describe which reactant and products react.

        # prediction unlabeled is unused
        prediction_unlabeled, prediction_labeled = drl.predict_concentration_euler(
            t_eval_pre=time_pre_addition,
            t_eval_post=time,
            initial_concentrations={"cat": 0.005 * 40 / 1200, "2": 0.005 * 800 / 1200},
            labeled_concentration={"2'": 0.005 * 800 / 2000},
            dilution_factor=1200 / 2000,
            # ivp_method='Radau',
            # dense_output=True,
            # rtol=1e-6,  # relative tolerance, kw in scipy.integrate.solve_ivp
        )

        # SYSTEM-SPECIFIC ENAMINE IONIZATION CORRECTION -> only a prediction of 4/5 can be made!
        for isomer in ISOMERS:
            for label in ["", "'"]:
                prediction_labeled = prediction_labeled.with_columns(
                    (pl.col(f"5{isomer}{label}") + ionization_factor * pl.col(f"4{isomer}{label}")).alias(
                        f"4/5{isomer}{label}")
                )

        predicted_total_compound = prediction_labeled[COMPOUND_RATIO[1]].sum(axis=1)
        predicted_compound_ratio = prediction_labeled[COMPOUND_RATIO[0]] / predicted_total_compound
        return prediction_labeled, predicted_compound_ratio[-100:].mean()

    @staticmethod
    def calculate_curves(data: pl.DataFrame) -> dict[str: pl.Series]:
        curves = {}
        for intermediate in INTERMEDIATES:
            # sum does not have to be recalculated between the isomer runs
            sum_all_isomers = data[[intermediate + isomer for isomer in ISOMERS]].sum(axis=1)
            for isomer in ISOMERS:
                # allows for easy modification of weight. str.contains('int_3') is much more specific than just '3'
                chemical_iso_split = f"int_{intermediate}_iso_{isomer}"
                chemical = f"{intermediate}{isomer}"  # 3D, 3E, 3F, 4/5D, 4/5E, 3/5F

                sum_chemical = data[[chemical, f"{chemical}'"]].sum(axis=1)

                curves[f"label_{chemical_iso_split}"] = data[chemical] / sum_chemical  # 3D / (3D+3D')
                curves[f"isomer_{chemical_iso_split}"] = data[chemical] / sum_all_isomers  # 3D / (3D+3E+3F)
                curves[f"TIC_{chemical_iso_split}"] = data[chemical] / sum_chemical[
                                                                       -100:].mean()  # normalized TIC curve
                curves[f"TIC_{chemical_iso_split}'"] = data[f"{chemical}'"] / sum_chemical[
                                                                              -100:].mean()  # normalized TIC curve
        return curves


RCO = RateConstantOptimizer(raw_weights=WEIGHTS, experimental=experimental, metric=METRIC)

# define the bounds, and vertex for the nelder-mead optimization
dimension_descriptions = rate_constant_names + ["ion"]

constraints = pd.DataFrame(np.full((3, len(dimension_descriptions)), np.nan), index=["vertex", "lower", "upper"],
                           columns=dimension_descriptions).T

index_reverse_reaction = constraints.index.str.contains("k-")
constraints[~index_reverse_reaction] = [1, 1e-6, 50]  # forwards; vertex, lb, ub
constraints[index_reverse_reaction] = [0.5, 0, 50]    # backwards

# special case
constraints[constraints.index.str.contains("ion")] = [0.01, 1e-6, 1]

# either chemically or experimentally determined to be zero
constraints[constraints.index.str.contains("k-1")] = [0, 0, 0]
constraints[constraints.index.str.contains("k-3")] = [0, 0, 0]
constraints[constraints.index.str.contains("k-4")] = [0, 0, 0]
constraints[constraints.index.str.contains("k2")] = [0, 0, 0]
vertex = constraints["vertex"].to_numpy()
bounds = [(lb, ub,) for _, (_, lb, ub) in constraints.iterrows()]

OPTIMIZE_TYPE = f"example_optimize_singular"  # in which path it should be saved (relative to a main folder)

path = f"./optimization/{OPTIMIZE_TYPE}/"
method_description = OPTIMIZE_TYPE.replace("_", " ")  # will be used as the plot title, defaults to OPTIMIZE_TYPE

optimize=True
if optimize:
    RCO.optimize(
        x0=vertex,
        x_description=dimension_descriptions,
        bounds=bounds,
        path=path,
        maxiter=200,
        _overwrite_log=True
    )
