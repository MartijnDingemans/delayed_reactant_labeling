from __future__ import annotations

from copy import deepcopy
import numpy as np
import pandas as pd

import sys
sys.path.insert(1, r'C:\Users\mdingemans\delayed_reactant_labeling\src')

from delayed_reactant_labeling.optimize import RateConstantOptimizerTemplate
from delayed_reactant_labeling.predict import DRL

EXPERIMENTAL_DATA_PATH = r'experimental_data_Roelant.xlsx'
CONCENTRATIONS_INITIAL = {"cat": 0.005 * 40 / 1200,  # concentration in M
                          "2": 0.005 * 800 / 1200}
CONCENTRATION_LABELED_REACTANT = {"2'": 0.005 * 800 / 2000}
DILUTION_FACTOR = 1200 / 2000

TIME_OF_ADDITION_COMPOUND = 0  # in minutes; start of the reaction
TIME_OF_ADDITION_LABELED_COMPOUND = 10.15  # in minutes; start of DRL curves

experimental_complete = pd.read_excel(EXPERIMENTAL_DATA_PATH, engine='openpyxl')
time_complete = experimental_complete["time (min)"]  # pre- and post-addition

labeled_chemicals = [chemical for chemical in experimental_complete.columns if chemical[-1] == "'"]
index_compound = np.argmax(
    experimental_complete["time (min)"] > TIME_OF_ADDITION_COMPOUND)  # first element of the post-addition situation
index_labeled_compound = np.argmax(experimental_complete[
                                       "time (min)"] > TIME_OF_ADDITION_LABELED_COMPOUND)  # first element of the post-addition situation

# correct for noise in intensity (y-axis) for the labeled chemicals!
for chemical in labeled_chemicals:
    experimental_complete[chemical] = experimental_complete[chemical] - experimental_complete[chemical].iloc[index_labeled_compound - 10:index_labeled_compound].median()


# only the situation post-addition of labeled compound is relevant for other parts of the script
experimental = experimental_complete.iloc[index_labeled_compound:, :]
time = experimental["time (min)"].to_numpy()
time_pre = experimental_complete.loc[:index_labeled_compound, 'time (min)'].to_numpy()

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
    def calculate_curves(data: pd.DataFrame) -> dict[str, np.ndarray]:
        curves = {}
        for intermediate in INTERMEDIATES:
            # sum does not have to be recalculated between the isomer runs
            sum_all_isomers = data[[f'{intermediate}{isomer}' for isomer in ISOMERS]].sum(axis=1)
            for isomer in ISOMERS:
                chemical = f"{intermediate}{isomer}"  # 3D, 3E, 3F, 4/5D, 4/5E, 3/5F
                chemical_iso_split = f"int_{intermediate}_iso_{isomer}"  # allows for easy modification of weight. str.contains('int_1') is much more specific than just '1'

                sum_chemical = data[[chemical, f"{chemical}'"]].sum(axis=1)

                curves[f"label_{chemical_iso_split}"] = (
                    data[chemical] / sum_chemical).to_numpy()  # 3D / (3D+3D')
                curves[f"isomer_{chemical_iso_split}"] = (
                    data[chemical] / sum_all_isomers).to_numpy()  # 3D / (3D+3E+3F)
                curves[f"TIC_{chemical_iso_split}"] = (
                        data[chemical] / sum_chemical[-100:].mean()).to_numpy()  # normalized TIC curve
                curves[f"TIC_{chemical_iso_split}'"] = (
                        data[f"{chemical}'"] / sum_chemical[-100:].mean()).to_numpy()  # normalized TIC curve
        return curves

    @staticmethod
    def create_prediction(x: np.ndarray, x_description: list[str]) -> pd.DataFrame:
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
        prediction_labeled = drl.predict_concentration(
            t_eval_pre=time_pre,
            t_eval_post=time,
            initial_concentrations=CONCENTRATIONS_INITIAL,
            labeled_concentration=CONCENTRATION_LABELED_REACTANT,
            dilution_factor=DILUTION_FACTOR,
            rtol=1e-8,
            atol=1e-8,
        )

        # SYSTEM-SPECIFIC ENAMINE IONIZATION CORRECTION -> only a prediction of 4/5 can be made!
        for isomer in ISOMERS:
            for label in ["", "'"]:
                prediction_labeled.loc[:, f"4/5{isomer}{label}"] = \
                    prediction_labeled.loc[:, f"5{isomer}{label}"] \
                    + ionization_factor * prediction_labeled.loc[:, f"4{isomer}{label}"]

        return prediction_labeled


RCO = RateConstantOptimizer(raw_weights=WEIGHTS, experimental=experimental, metric=METRIC)

# the rate constant optimizer class is independent of your predicted run.
x_description = rate_constant_names + ['ion']
constraints = pd.DataFrame(np.full((3, len(x_description)), np.nan), index=["vertex", "lower", "upper"],
                           columns=x_description).T

index_reverse_reaction = constraints.index.str.contains("k-")
constraints.iloc[np.nonzero(~index_reverse_reaction)] = [1, 1e-6, 1e3]  # forwards; vertex, lb, ub
constraints.iloc[np.nonzero(index_reverse_reaction)] = [0.5, 0, 1e3]  # backwards

# special case
constraints.iloc[np.nonzero(constraints.index.str.contains("ion"))] = [0.01, 1e-6, 1]

constraints.iloc[np.nonzero(constraints.index.str.contains("k2_D"))] = [0.410972, 0.3789, 0.4530]
constraints.iloc[np.nonzero(constraints.index.str.contains("k2_E"))] = [0.644027, 0.5227, 0.8002]
constraints.iloc[np.nonzero(constraints.index.str.contains("k2_F"))] = [0.510573, 0.2763, 1.1920]

# either chemically or experimentally determined to be zero
constraints.iloc[np.nonzero(constraints.index.str.contains("k-1"))] = [0, 0, 0]
constraints.iloc[np.nonzero(constraints.index.str.contains("k-3"))] = [0, 0, 0]
constraints.iloc[np.nonzero(constraints.index.str.contains("k-4"))] = [0, 0, 0]
bounds = [(lb, ub,) for _, (_, lb, ub) in constraints.iterrows()]

RCO.optimize_multiple(
    path=r'./optimize/',
    n_runs=64,
    x_description=x_description,
    x_bounds=bounds,
    n_jobs=16,
    maxiter=2000,
)
