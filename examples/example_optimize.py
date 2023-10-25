import math
import os
from copy import deepcopy

import pandas as pd
import polars as pl  # just much more efficient than pandas, although not as easy to use.
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from icecream import ic

from delayed_reactant_labeling.predict_new import DRL
from delayed_reactant_labeling.optimize import RateConstantOptimizerTemplate
from delayed_reactant_labeling.visualize import VisualizeSingleSolution, VisualizeMultipleSolutions

#%%
experimental_complete = pl.read_excel("experimental_data_Roelant.xlsx", engine='openpyxl')
index_labeled_compound = np.argmax(experimental_complete["time (min)"] > 10.15)

time_pre_addition = experimental_complete['time (min)'][:index_labeled_compound]

# only look at the part after the addition of labeled compound
experimental = experimental_complete[index_labeled_compound:, :]
time = experimental['time (min)']


#%%
WEIGHT_TIME = 1 - 0.9 * np.linspace(0, 1, time.shape[0])  # decrease weight with time, first point 10 times as import as last point
WEIGHT_TIME = WEIGHT_TIME / sum(WEIGHT_TIME)  # normalize


def METRIC(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.average(np.abs(y_pred - y_true), weights=WEIGHT_TIME, axis=0)


#%%

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


#%%
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
        prediction_unlabeled, prediction_labeled = drl.predict_concentration(
            t_eval_pre=time_pre_addition,
            t_eval_post=time,
            initial_concentrations={"cat": 0.005 * 40 / 1200, "2": 0.005 * 800 / 1200},
            labeled_concentration={"2'": 0.005 * 800 / 2000},
            dilution_factor=1200 / 2000,
            ivp_method='Radau',
            dense_output=True,
            rtol=1e-6,  # relative tolerance, kw in scipy.integrate.solve_ivp
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


#%%
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

# create the prediction, calculate the error, and weigh them. This exact procedure will also be followed when optimizing the entire system.
labeled_prediction = RCO.create_prediction(x=x, x_description=x_description)[0]  # prediction

errors = RCO.calculate_error_functions(labeled_prediction)
weighed_errors = RCO.weigh_errors(errors)

df = pd.DataFrame([errors, weighed_errors], index=["normal", "weighed"]).T
print(df)
ic(weighed_errors.sum())

fig, ax = plt.subplots(tight_layout=True)
weighed_errors.T.plot.bar(ax=ax)
ax.set_ylabel("MAE")


#%%

fig_label, axs_label = plt.subplots(3, 1, tight_layout=True, figsize=(8, 8), squeeze=False)
fig_isomer, axs_isomer = plt.subplots(2, 1, tight_layout=True, squeeze=False)
fig_TIC, axs_TIC = plt.subplots(3, 2, tight_layout=True, figsize=(8, 8), squeeze=False)
marker_settings = {"alpha": 0.4, "marker": ".", "s": 1}

true = RCO.experimental_curves
pred = RCO.calculate_curves(labeled_prediction)

for i, intermediate in enumerate(INTERMEDIATES):
    # sum does not have to be recalculated between the isomer runs
    sum_all_isomers = labeled_prediction[[intermediate + isomer for isomer in ISOMERS]].sum(axis=1)
    for j, isomer in enumerate(ISOMERS):
        # the "iso_" prefix is given to each chemical so that we can search the strings for e.g. "iso_A" and not get a match for label
        chemical_iso_split = f"int_{intermediate}_iso_{isomer}"

        # plot label ratio
        axs_label[j, 0].plot(time, pred[f"label_{chemical_iso_split}"], color=f"C{i}",
                             label=f"{chemical_iso_split} MAE: {errors[f'label_{chemical_iso_split}']:.3f}")
        axs_label[j, 0].scatter(time, true[f"label_{chemical_iso_split}"], color=f"C{i}", **marker_settings)
        # the curve of the labeled compound is the same, by definition, as 1 - unlabeled
        axs_label[j, 0].plot(time, 1 - pred[f"label_{chemical_iso_split}"], color="tab:gray")
        axs_label[j, 0].scatter(time, 1 - true[f"label_{chemical_iso_split}"], color="tab:gray", **marker_settings)

        # isomer ratio
        axs_isomer[i, 0].plot(time, pred[f"isomer_{chemical_iso_split}"],
                              label=f"{chemical_iso_split} MAE: {errors[f'isomer_{chemical_iso_split}']:.3f}")
        axs_isomer[i, 0].scatter(time, RCO.experimental_curves[f"isomer_{chemical_iso_split}"], **marker_settings)

        # TIC shape
        axs_TIC[j, i].plot(time, pred[f"TIC_{chemical_iso_split}"],
                           color="tab:blue",
                           label=f"{chemical_iso_split} MAE: {errors[f'TIC_{chemical_iso_split}']:.3f}")
        axs_TIC[j, i].scatter(time, RCO.experimental_curves[f"TIC_{chemical_iso_split}"], color="tab:blue",
                              **marker_settings)

        axs_TIC[j, i].plot(time, pred[f"TIC_{chemical_iso_split}'"],
                           color="tab:gray",
                           label=f"""{chemical_iso_split} MAE: {errors[f"TIC_{chemical_iso_split}'"]:.3f}""")
        axs_TIC[j, i].scatter(time, RCO.experimental_curves[f"TIC_{chemical_iso_split}'"], color="tab:gray",
                              **marker_settings)

fig_isomer.supylabel("isomer ratio")
fig_isomer.supxlabel("time (min)")
for ax in axs_isomer.flatten():
    ax.legend()
fig_isomer.show()

fig_label.supylabel("labeled ratio")
fig_label.supxlabel("time (min)")
for ax in axs_label.flatten():
    ax.legend()
fig_label.show()

fig_TIC.supylabel("normalized TIC")
fig_TIC.supxlabel("time (min)")
for ax in axs_TIC.flatten():
    ax.legend()
fig_TIC.show()




