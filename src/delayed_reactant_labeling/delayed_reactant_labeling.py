import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error
from dataclasses import dataclass
from typing import Optional
from numba import njit
from numba.typed import List
from copy import deepcopy
import warnings


@dataclass
class Experimental_Conditions:
    time: tuple[np.ndarray, np.ndarray]
    initial_concentrations: dict[str, float]
    dilution_factor: float
    labeled_reactant: dict[str, float]
    mass_balance: Optional[list[str]] = None

    def copy(self):
        return deepcopy(self)


@njit
def calculate_step(reaction_rate, reaction_reactants, reaction_products, delta_time, concentrations: np.ndarray):
    new_concentration = concentrations.copy()
    for i in range(reaction_rate.shape[0]):
        created_amount = delta_time * reaction_rate[i] * np.prod(concentrations[reaction_reactants[i]])
        new_concentration[reaction_reactants[i]] -= created_amount  # consumed
        new_concentration[reaction_products[i]] += created_amount  # produced
    return new_concentration


class DRL:
    def __init__(self,
                 reactions: list[tuple[str, list[str], list[str]]],
                 rates: dict[str: float]):

        # link the name of a chemical with an index
        self.reference = set()
        for k, reactants, products in reactions:
            for compound in reactants + products:
                self.reference.add(compound)
        self.reference = {compound: n for n, compound in enumerate(sorted(self.reference))}
        self.initial_concentrations = np.zeros((len(self.reference)))

        # store the last used time slice
        self.time = None

        # construct the list of indexes of reactants and products per reaction
        self.rate_equations = []  # only for us mere humans
        self.reaction_rate = []  # np array at the end
        self.reaction_reactants = List()  # multiply everything per reaction, and multiply by k
        self.reaction_products = List()  # add

        for k, reactants, products in reactions:
            if rates[k] == 0:
                # the reaction does not create or consume any chemicals, therefore its redundant and can be removed for
                # computational benefits
                continue

            # human-readable string, machine executable function
            self.rate_equations.append(f"{k}*" + "*".join([f"[{reactant}]" for reactant in reactants]))
            self.reaction_rate.append(rates[k])
            self.reaction_reactants.append(np.array([self.reference[reactant] for reactant in reactants]))
            self.reaction_products.append(np.array([self.reference[product] for product in products]))
        self.reaction_rate = np.array(self.reaction_rate)

    def predict_concentration_slice(self, initial_concentration: np.ndarray, time_slice: np.ndarray, mass_balance):
        # allows the format last slice function to perform formatting based on the last used time_slice
        self.time = time_slice

        prev_prediction = initial_concentration
        predicted_concentration = np.full((len(time_slice), len(initial_concentration)), np.nan)
        predicted_concentration[0, :] = initial_concentration

        # for the first step more steps are required; as by definition, no 5 or 6 could be formed in a singular step.
        prev_t = time_slice[0]
        new_prediction = None
        for new_t in np.linspace(prev_t, time_slice[1], 30)[1:]:
            new_prediction = calculate_step(
                reaction_rate=self.reaction_rate,
                reaction_reactants=self.reaction_reactants,
                reaction_products=self.reaction_products,
                concentrations=prev_prediction,
                delta_time=new_t - prev_t)

            prev_t = new_t
            prev_prediction = new_prediction
        predicted_concentration[1, :] = new_prediction

        # use the given steps
        for row, new_t in enumerate(time_slice[2:]):
            new_prediction = calculate_step(
                reaction_rate=self.reaction_rate,
                reaction_reactants=self.reaction_reactants,
                reaction_products=self.reaction_products,
                concentrations=prev_prediction,
                delta_time=new_t - prev_t, )

            predicted_concentration[row + 2, :] = new_prediction
            prev_t = new_t
            prev_prediction = new_prediction
        df_result = pd.DataFrame(predicted_concentration, columns=list(self.reference.keys()))
        df_result["time (min)"] = time_slice
        last_prediction = prev_prediction

        if mass_balance is not None:
            mass_sum = np.sum(predicted_concentration[:, [self.reference[chemical] for chemical in mass_balance]],
                              axis=1)
            if not all(mass_sum - mass_sum[0] < 1e-14):
                raise ValueError("The mass balance was not obeyed.")

        return df_result, last_prediction

    def predict_concentration(self,
                              exp_condition: Experimental_Conditions
                              ) -> (pd.DataFrame, pd.DataFrame):

        # reorder the initial concentrations such that they match with the sorting in self.reference
        for compound, initial_concentration in exp_condition.initial_concentrations.items():
            self.initial_concentrations[self.reference[compound]] = initial_concentration

        # pre addition
        result_pre_addition, last_prediction = self.predict_concentration_slice(
            initial_concentration=self.initial_concentrations,
            time_slice=exp_condition.time[0],
            mass_balance=exp_condition.mass_balance
        )

        # dillution step
        diluted_concentrations = last_prediction * exp_condition.dilution_factor
        for reactant, concentration in exp_condition.labeled_reactant.items():
            diluted_concentrations[self.reference[reactant]] = concentration

        # post addition
        results_post_addition, _ = self.predict_concentration_slice(
            initial_concentration=diluted_concentrations,
            time_slice=exp_condition.time[1],
            mass_balance=exp_condition.mass_balance
        )
        return result_pre_addition, results_post_addition

    def format_last_slice(self, experimental: pd.DataFrame, prediction: pd.DataFrame):
        """
        format the last slice of the predicted and experimental data such that they both start at t=0,
        and have the same size
        """
        if self.time is None:
            raise ValueError("No time parameter has been defined.")

        # prepare the size of each dataset
        _experimental = experimental.copy()
        _experimental = _experimental[_experimental["time (min)"].between(self.time[0], self.time[-1])].reset_index(
            drop=True)
        _experimental["time (min)"] = _experimental["time (min)"] - _experimental["time (min)"][0]

        _prediction = prediction.copy()
        _prediction["time (min)"] = _prediction["time (min)"] - _prediction["time (min)"][0]

        return _experimental, _prediction


def main():
    """Compare the results found using the DRL class with the original excel sheet from Roelant et al."""

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

    def create_reactions_twoway():
        """
        Create the reverse reaction of each pre-defined reaction.

        Removes the first character of each rate constant name (should be "k"), and adds to the list of reactions a new reaction with "k-" in front of the name, with the products and reactants reversed."""
        reactions_twoway_labeled = deepcopy(REACTIONS_ONEWAY)
        for k, reactants, products in REACTIONS_ONEWAY:
            reactions_twoway_labeled.append(("k-" + k[1:], products, reactants))

        return reactions_twoway_labeled

    reaction_equations = create_reactions_twoway()

    EXPERIMENTAL_DATA_PATH = "../../experimental_data_Roelant.xlsx"  # the absolute path can also be given
    experimental = pd.read_excel(EXPERIMENTAL_DATA_PATH)
    time = experimental["time (min)"]
    assert time.is_monotonic_increasing

    CONCENTRATIONS_INITIAL = {"cat": 0.005 * 40 / 1200,  # concentration in M
                              "2": 0.005 * 800 / 1200}
    CONCENTRATION_LABELED_REACTANT = {"2'": 0.005 * 800 / 2000}
    DILLUTION_FACTOR = 1200 / 2000

    TIME_OF_ADDITION_LABELED_COMPOUND = 10.  # in minutes; start of DRL curves
    index_labeled_compound = np.argmax(experimental["time (min)"] > TIME_OF_ADDITION_LABELED_COMPOUND)

    experimental_conditions = Experimental_Conditions(
        time=(experimental["time (min)"].iloc[:index_labeled_compound].to_numpy(),
              experimental["time (min)"].iloc[index_labeled_compound:].to_numpy()),
        initial_concentrations=CONCENTRATIONS_INITIAL,
        dilution_factor=DILLUTION_FACTOR,
        labeled_reactant=CONCENTRATION_LABELED_REACTANT,
    )

    rate_constants = {
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
    rate_constants = pd.Series(rate_constants)

    from timeit import default_timer

    t = default_timer()
    for i in range(1):
        drl = DRL(reactions=reaction_equations,
                  rates=rate_constants)
        _prediction_unlabeled, _prediction_labeled = drl.predict_concentration(experimental_conditions)
    print(f"this took {default_timer()-t:.2f} seconds")

    def verify_results(data_me: pd.DataFrame,
                       data_original: pd.DataFrame,
                       metric: any,
                       skip_label: bool = False):

        data_original = data_original.copy()
        data_original.columns = [str(column) for column in data_original.columns]
        data_original["time (min)"] = data_original["time (min)"] - data_original["time (min)"][0]

        _diff = {}
        for compound in data_me:
            if skip_label is True and compound[-1] == "'":
                continue
            elif compound[:2] == "6F":
                continue
            elif compound == "time (min)":
                continue
            y_true = data_original[compound]
            y_pred = data_me[compound]
            error = metric(y_true=y_true[:-1], y_pred=y_pred)
            print(f"{compound} has a {metric.__name__} of \t{error:.6e}")
            y = (y_pred - y_true) / y_true * 100
            ax.plot(y, label=compound)
        ax.set_ylabel("relative error\n(Ypython - Yroelant)/Yroelant*100 (%)")
        ax.legend()
        ax.set_title("prediction Roelant - prediction python")
        fig.show()

    print("------------pre-------------")
    fig, ax = plt.subplots(1, 1, figsize=(8.5, 11.5))
    verify_results(
        data_me=_prediction_unlabeled,
        data_original=pd.read_excel("../../predicted_DRL_ORIGINAL_pre_addition.xlsx"),
        skip_label=True,
        metric=mean_absolute_error
    )
    print("------------POST------------")
    fig, ax = plt.subplots(1, 1, figsize=(8.5, 11.5))
    verify_results(
        data_me=_prediction_labeled,
        data_original=pd.read_excel("../../predicted_DRL_ORIGINAL_post_addition.xlsx"),
        skip_label=False,
        metric=mean_absolute_error
    )


if __name__ == "__main__":
    main()
