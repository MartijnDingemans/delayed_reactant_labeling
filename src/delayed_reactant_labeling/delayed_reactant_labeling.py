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


"""
def main():
    # compare our results with Roelant
    import drl_utilities.reaction_roelant as reaction_roelant
    path = "../experiments/MAH174/labeled_aldehyde_Roelant_data/images/original_rate_constants/numba_attempt_"

    drl = DRL(reactions=reaction_roelant.reactions_twoway,
              rates=reaction_roelant.rate_constants)

    # use the same time span as he did for the predictions!
    roelant_prediction_pre_addition = pd.read_excel("E:/DRL/show_case/predicted_DRL_ORIGINAL_pre_addition.xlsx")[
        "time (min)"].to_numpy()
    roelant_prediction_post_addition = pd.read_excel("E:/DRL/show_case/predicted_DRL_ORIGINAL_post_addition.xlsx")[
        "time (min)"].to_numpy()

    exp_conditions = Experimental_Conditions(
        time=(roelant_prediction_pre_addition, roelant_prediction_post_addition,),
        initial_concentrations=reaction_roelant.initial_concentration,
        dilution_factor=1200 / 2000,
        labeled_reactant={"2'": 0.005 * 800 / 2000},
    )
    _prediction_unlabeled, _prediction_labeled = drl.predict_concentration(exp_conditions)

    def verify_results(data_me: pd.DataFrame,
                       data_original: pd.DataFrame,
                       metric: any,
                       skip_label: bool = False):
        compounds = {"cat": "1", "2": "2", "2'": "2'", "6B": "R-6", "6B'": "R-6'", "6C": "S-6", "6C'": "S-6'"}
        for p in ["A", "B", "C"]:
            for intermediate in ["3", "4", "5", "5"]:
                for label in ["", "'"]:
                    compounds[f"{intermediate}{p}{label}"] = f"{intermediate}{p}{label}"

        data_original = data_original.copy()
        data_original.columns = [str(col).strip() for col in data_original.columns]
        data_original["time (min)"] = data_original["time (min)"] - data_original["time (min)"][0]

        _diff = {}
        for compound_me, compound_they in compounds.items():
            if skip_label is True and compound_me[-1] == "'":
                continue
            y_true = data_original[compound_they]
            y_pred = data_me[compound_me]
            error = metric(y_true=y_true, y_pred=y_pred)
            print(f"{compound_me} has a {metric.__name__} of {error:.6f}")
            y = (y_pred - y_true) / y_true * 100
            ax.plot(y, label=compound_me)
        ax.legend()
        ax.set_title("prediction Roelant - prediction python")
        fig.show()
        fig.savefig(f"{path}verify_results_label_{not skip_label}.png", dpi=1000)

    print("------------pre-------------")
    fig, ax = plt.subplots(1, 1, figsize=(8.5, 11.5))
    verify_results(
        data_me=_prediction_unlabeled,
        data_original=pd.read_excel("E:/DRL/show_case/predicted_DRL_ORIGINAL_pre_addition.xlsx"),
        skip_label=True,
        metric=mean_absolute_error
    )
    print("------------POST------------")
    fig, ax = plt.subplots(1, 1, figsize=(8.5, 11.5))
    verify_results(
        data_me=_prediction_labeled,
        data_original=pd.read_excel("E:/DRL/show_case/predicted_DRL_ORIGINAL_post_addition.xlsx"),
        skip_label=False,
        metric=mean_absolute_error
    )

    # load experimental data, make sure it is aligned correctly.
    experimental = pd.read_excel("E:/DRL/show_case/ORIGINAL_experimental_data.xlsx")
    experimental.columns = [str(col).strip().upper() for col in experimental.columns]
    experimental.rename(columns={"TIME (MIN)": "time (min)"}, inplace=True)
    experimental["time (min)"] = experimental["time (min)"] + roelant_prediction_pre_addition[-1]

    # repeat the DRL prediction step, this time utilizing the same time steps as the measured spectra frames
    # in the Excel sheet the two time steps do NOT align. Try to work around it as much as possible.
    exp_conditions.time = (roelant_prediction_pre_addition, experimental["time (min)"].to_numpy())

    _prediction_unlabeled, _prediction_labeled = drl.predict_concentration(exp_conditions)
    experimental, _prediction_labeled = drl.format_last_slice(experimental, _prediction_labeled)

    print("calculating errors p2")
    errors, (fig_iso, axs_iso), (fig_lab, axs_lab) = reaction_roelant.compare_data(
        experimental,
        _prediction_labeled,
        mean_absolute_error,
        plot_label_ratio=True,
        plot_isomer_ratio=True)

    errors_at_5min = reaction_roelant.compare_data(experimental[experimental["time (min)"].between(0, 5)],
                                                   _prediction_labeled[_prediction_labeled["time (min)"].between(0, 5)],
                                                   mean_absolute_error)
    # replace np.inf with a large value
    for key, value in errors.items():
        if np.isinf(value):
            errors[key] = 1e3
    for key, value in errors_at_5min.items():
        if np.isinf(value):
            errors_at_5min[key] = 1e3

    # show a bar plot of all error contributions
    error_all = pd.DataFrame([errors, errors_at_5min], index=["all", "first 5 minutes"])
    descr = f"original rate constants, sum(MAE) = {np.sum(error_all.sum().sum()):.2f}"
    fig, ax = plt.subplots()
    error_all.T.plot.bar(ax=ax)
    ax.set_title(descr)
    ax.set_xlabel("error type")
    ax.set_ylabel("MAE")
    fig.tight_layout()
    fig.show()
    fig.savefig(f"{path}bar_plot_of_all_errors.png", dpi=1000)

    fig_lab.tight_layout()
    fig_lab.show()
    for ax in axs_lab:
        ax.legend()
    fig_lab.savefig(f"{path}_fit_with_labeled.png", dpi=1000)

    axs_iso[0].legend()
    axs_iso[1].legend()
    fig_iso.tight_layout()
    fig_iso.show()
    fig_iso.savefig(f"{path}_fit_between_isomers.png", dpi=1000)

    time = experimental["time (min)"]
    fig, axs = plt.subplots(3, 2, figsize=(11, 11))
    for axs_row, isomer in zip(axs, ["A", "B", "C"]):
        for ax, compound in zip(axs_row, ["3", "5"]):
            ax.scatter(time, experimental[f"{compound}{isomer}"], color="tab:blue", alpha=0.3, s=1)
            ax.scatter(time, experimental[f"{compound}{isomer}'"], color="tab:gray", alpha=0.3, s=1)
            ax.set_ylabel("experimental intensity (a.u.)")
            ax.set_xlabel("time (min)")
            ax.set_xlim(left=0.2)
            ax.set_ylim(bottom=0)
            height_exp = experimental[f"{compound}{isomer}"].iloc[200:].mean()

            ax2 = ax.twinx()
            if compound == "3":
                pred = _prediction_labeled[f"3{isomer}"]
                ax2.plot(time, _prediction_labeled[f"3{isomer}"], color="tab:blue", label=f"3{isomer}")
                ax2.plot(time, _prediction_labeled[f"3{isomer}'"], color="tab:gray", label=f"3{isomer}'")
            else:
                pred = 0.025 * _prediction_labeled[f"4{isomer}"] + _prediction_labeled[f"5{isomer}"]
                ax2.plot(time, 0.025 * _prediction_labeled[f"4{isomer}"] + _prediction_labeled[f"5{isomer}"],
                         color="tab:blue",
                         label=f"4{isomer} / 5{isomer}")
                ax2.plot(time, 0.025 * _prediction_labeled[f"5{isomer}'"] + _prediction_labeled[f"5{isomer}'"],
                         color="tab:gray", label=f"4{isomer}' / 5{isomer}'")

            yl, yu = ax2.get_ylim()
            height_pred = pred.iloc[200:].mean()
            ax2.set_ylim(bottom=0, top=yu * ((height_pred / yu) / (height_exp / ax.get_ylim()[1])))
            ax.set_title(f"intensity / M = {height_exp / height_pred:.2e}")

            ax2.legend()
            ax2.set_ylabel("predicted concentration (M)")
    fig.tight_layout()
    fig.show()
    fig.savefig(f"{path}intensities_over_time")


if __name__ == "__main__":
    main()
    print("done and gone")
"""