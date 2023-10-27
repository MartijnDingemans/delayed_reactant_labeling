from copy import deepcopy

import numpy as np
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt

from icecream import ic
from src.delayed_reactant_labeling.predict_new import DRL
from src.delayed_reactant_labeling.optimize import RateConstantOptimizerTemplate


EXPERIMENTAL_DATA_PATH = r'C:\Users\mdingemans\delayed_reactant_labeling\tools\experimental_data\experimental_data_Roelant.xlsx'  # the absolute path can also be given
CONCENTRATIONS_INITIAL = {"cat": 0.005 * 40 / 1200,  # concentration in M
                          "2": 0.005 * 800 / 1200}
CONCENTRATION_LABELED_REACTANT = {"2'": 0.005 * 800 / 2000}
DILUTION_FACTOR = 1200 / 2000

TIME_OF_ADDITION_COMPOUND = 0  # in minutes; start of the reaction
TIME_OF_ADDITION_LABELED_COMPOUND = 10.15  # in minutes; start of DRL curves

experimental_complete = pl.read_excel(EXPERIMENTAL_DATA_PATH, engine='openpyxl')
time_complete = experimental_complete["time (min)"]  # pre- and post-addition

labeled_chemicals = [chemical for chemical in experimental_complete.columns if chemical[-1] == "'"]
index_compound = np.argmax(
    experimental_complete["time (min)"] > TIME_OF_ADDITION_COMPOUND)  # first element of the post-addition situation
index_labeled_compound = np.argmax(experimental_complete[
                                       "time (min)"] > TIME_OF_ADDITION_LABELED_COMPOUND)  # first element of the post-addition situation

# correct for noise in intensity (y-axis) for the labeled chemicals!
experimental_complete = experimental_complete.with_columns(
    [(pl.col(chemical) - pl.col(chemical).slice(index_labeled_compound - 10, 10).median()).alias(chemical) for
     chemical in labeled_chemicals]
)

# only the situation post-addition of labeled compound is relevant for other parts of the script
experimental = experimental_complete[index_labeled_compound:, :]
time = experimental["time (min)"].to_numpy()
time_pre = experimental_complete['time (min)'][:index_labeled_compound].to_numpy()

WEIGHT_TIME = 1 - 0.9 * np.linspace(0, 1, time.shape[
    0])  # decrease weight with time, first point 10 times as import as last point
WEIGHT_TIME = WEIGHT_TIME / sum(WEIGHT_TIME)  # normalize


def METRIC(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    :param y_true: experimental data
    :param y_pred: data from prediction
    :return: error
    """
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

# These two groups will allow us to simplify the code later on. Allows for filtering per isomer or intermediate.
# 4 and 5 are grouped as 4/5 as all the enamine 4 will be converted to 5 during the ionization process in the mass spectrometer.
INTERMEDIATES = ["3", "4/5"]
ISOMERS = ["D", "E", "F"]


def create_reaction_equations():
    """
    Create the reverse reaction of each pre-defined reaction.

    Removes the first character of each rate constant name (should be "k"), and adds to the list of reactions a new reaction with "k-" in front of the name, with the products and reactants reversed."""
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
    def calculate_curves(data: pl.DataFrame) -> dict[str: pl.Series]:
        curves = {}
        for intermediate in INTERMEDIATES:
            # sum does not have to be recalculated between the isomer runs
            sum_all_isomers = data[[intermediate + isomer for isomer in ISOMERS]].sum(axis=1)
            for isomer in ISOMERS:
                chemical = f"{intermediate}{isomer}"  # 3D, 3E, 3F, 4/5D, 4/5E, 3/5F
                chemical_iso_split = f"int_{intermediate}_iso_{isomer}"  # allows for easy modification of weight. str.contains('int_1') is much more specific than just '1'

                sum_chemical = data[[chemical, f"{chemical}'"]].sum(axis=1)

                curves[f"label_{chemical_iso_split}"] = data[chemical] / sum_chemical  # 3D / (3D+3D')
                curves[f"isomer_{chemical_iso_split}"] = data[chemical] / sum_all_isomers  # 3D / (3D+3E+3F)
                curves[f"TIC_{chemical_iso_split}"] = data[chemical] / sum_chemical[
                                                                       -100:].mean()  # normalized TIC curve
                curves[f"TIC_{chemical_iso_split}'"] = data[f"{chemical}'"] / sum_chemical[
                                                                              -100:].mean()  # normalized TIC curve
        return curves

    @staticmethod
    def create_prediction(x: np.ndarray, x_description: list[str]) -> tuple[pl.DataFrame, float]:
        # separate out the ionization factor from the other parameters which are being optimized.
        rate_constants = pd.Series(x[:len(rate_constant_names)], index=x_description[:len(rate_constant_names)])
        ionization_factor = x[-1]

        rate_constants["k2_D"] = 0.410972
        rate_constants["k2_E"] = 0.644027
        rate_constants["k2_F"] = 0.510573

        drl = DRL(reactions=reaction_equations,
                  rate_constants=rate_constants,
                  verbose=False)  # stores values in drl.reactions which describe which reactant and products react.

        # prediction unlabeled is unused
        prediction_unlabeled, prediction_labeled = drl.predict_concentration(
            t_eval_pre=time_pre,
            t_eval_post=time,
            initial_concentrations=CONCENTRATIONS_INITIAL,
            labeled_concentration=CONCENTRATION_LABELED_REACTANT,
            dilution_factor=DILUTION_FACTOR,
            rtol=1e-6,
            atol=1e-6,
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


RCO = RateConstantOptimizer(raw_weights=WEIGHTS, experimental=experimental, metric=METRIC)


# these rate constants were found by Roelant et al. and are used as a example only
rate_constants_roelant = {
    "k1_D": 1.5,
    "k1_E": 0.25,
    "k1_F": 0.01,
    "k2_D": 0.43,
    "k2_E": 0.638,
    "k2_F": 0.567,
    "k3_D": 0.23,
    "k3_E": 0.35,
    "k3_F": 0.3,
    "k4_D": 8,
    "k4_E": 0.05,
    "k4_F": 0.03,
    "k-1_D": 0,
    "k-1_E": 0,
    "k-1_F": 0,
    "k-2_D": 0.025,
    "k-2_E": 0.035,
    "k-2_F": 0.03,
    "k-3_D": 0,
    "k-3_E": 0,
    "k-3_F": 0,
    "k-4_D": 0,
    "k-4_E": 0,
    "k-4_F": 0,
}

# define your inputs
x = np.array(list(rate_constants_roelant.values()) + [0.025])
x_description = list(rate_constants_roelant.keys()) + ['ion']
labeled_prediction = RCO.create_prediction(x=x, x_description=x_description)[0]  # prediction
errors = RCO.calculate_error_functions(labeled_prediction)
weighed_errors = RCO.weigh_errors(errors)

df = pd.DataFrame([errors, weighed_errors], index=["normal", "weighed"]).T
print(df)
ic(weighed_errors.sum())


# the rate constant optimizer class is independent of your predicted run.
x_description = rate_constant_names + ['ion']
constraints = pd.DataFrame(np.full((3, len(x_description)), np.nan), index=["vertex", "lower", "upper"],
                           columns=x_description).T

index_reverse_reaction = constraints.index.str.contains("k-")
constraints[~index_reverse_reaction] = [1, 1e-6, 1e3]  # forwards; vertex, lb, ub
constraints[index_reverse_reaction] = [0.5, 0, 1e3]  # backwards

# special case
constraints[constraints.index.str.contains("ion")] = [0.01, 1e-6, 1]

constraints[constraints.index.str.contains("k2_D")] = [0.410972, 0.3789, 0.4530]
constraints[constraints.index.str.contains("k2_E")] = [0.644027, 0.5227, 0.8002]
constraints[constraints.index.str.contains("k2_F")] = [0.510573, 0.2763, 1.1920]

# either chemically or experimentally determined to be zero
constraints[constraints.index.str.contains("k-1")] = [0, 0, 0]
constraints[constraints.index.str.contains("k-3")] = [0, 0, 0]
constraints[constraints.index.str.contains("k-4")] = [0, 0, 0]
bounds = [(lb, ub,) for _, (_, lb, ub) in constraints.iterrows()]

'''
vertex = [
    0.000000,
    0.000000,
    0.000000,
    285.801380,
    53.930702,
    383.368881,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    434.947552,
    974.186193,
    897.677608,
    0.410972,
    0.644027,
    0.510573,
    676.689352,
    60.802713,
    555.596117,
    271.451605,
    879.651173,
    64.214437,
    0.01
]
vertex = np.array(vertex)
'''

vertex = constraints['vertex'].to_numpy()

RCO.optimize(path=r'C:\Users\mdingemans\delayed_reactant_labeling\tools\optimize',
             x0=vertex,
             bounds=bounds,
             x_description=x_description, maxiter=200,
             _overwrite_log=True)

