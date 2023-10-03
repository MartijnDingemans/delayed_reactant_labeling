import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional
from numba import njit
from numba.typed import List
from copy import deepcopy


@dataclass
class Experimental_Conditions:
    """Class which stores the basic information which is required to simulate a reaction."""
    time: tuple[np.ndarray, np.ndarray]
    initial_concentrations: dict[str, float]
    dilution_factor: float
    labeled_reactant: dict[str, float]
    mass_balance: Optional[list[str]] = None

    def copy(self):
        return deepcopy(self)

    def __post_init__(self):
        """Check the elements of the time array, to prevent pd.Series objects being passed through."""
        for time_slice in self.time:
            if not isinstance(time_slice, np.ndarray):
                raise ValueError(f"Time slices must be np.ndarray but instead a {type(time_slice)} was found.")


@njit
def calculate_step(reaction_rate: np.ndarray,
                   reaction_reactants: List[np.ndarray],
                   reaction_products: List[np.ndarray], delta_time: float, concentrations: np.ndarray):
    """
    Calculates a singular step using the Explicit Euler formula.
    Foreach defined reaction all reactants will be decreased by the 'created amount',
    whereas the products will be increased.
    :param reaction_rate: Each element contains the rate constant values.
    :param reaction_reactants: Each element contains an array of the indices of which chemicals are the reactants.
    :param reaction_products: Each element contains an array of the indices of which chemicals are the products.
    :param delta_time: The time step which is to be simulated.
    :param concentrations: The current concentrations of each chemical.
    :return: The predicted concentrations.
    """
    new_concentration = concentrations.copy()
    for i in range(reaction_rate.shape[0]):
        created_amount = delta_time * reaction_rate[i] * np.prod(concentrations[reaction_reactants[i]])
        new_concentration[reaction_reactants[i]] -= created_amount  # consumed
        new_concentration[reaction_products[i]] += created_amount  # produced
    return new_concentration


class DRL:
    """Class which enables efficient prediction of changes in concentration in a chemical system.
    Especially useful for Delayed Reactant Labeling (DRL) experiments."""
    def __init__(self,
                 reactions: list[tuple[str, list[str], list[str]]],
                 rate_constants: dict[str: float]):
        """Initialize the chemical system.
        :param reactions: List of reactions, each reaction is given as a tuple.
        Its first element is a string, which determines which rate constant is applicable to that reaction.
        Its second element is a list containing the identifiers (strings) of each reactant in the reaction.
        The third element contains a list for the reaction products
        :param rate_constants: A dictionairy or which maps the rate constants to their respective values.
        """
        # link the name of a chemical with an index
        self.reference = set()
        for k, reactants, products in reactions:
            for compound in reactants + products:
                self.reference.add(compound)
        self.reference = {compound: n for n, compound in enumerate(sorted(self.reference))}
        self.initial_concentrations = np.zeros((len(self.reference)))

        # store the last used time slice
        self.time = None

        # construct a list containing the indices of all the reactants and products per reaction
        self.reaction_rate = []  # np array at the end
        self.reaction_reactants = List()  # multiply everything per reaction, and multiply by k
        self.reaction_products = List()  # add

        for k, reactants, products in reactions:
            if rate_constants[k] == 0:
                # the reaction does not create or consume any chemicals, therefore its redundant and can be removed for
                # computational benefits
                continue

            # human-readable string, machine executable function
            self.reaction_rate.append(rate_constants[k])
            self.reaction_reactants.append(np.array([self.reference[reactant] for reactant in reactants]))
            self.reaction_products.append(np.array([self.reference[product] for product in products]))
        self.reaction_rate = np.array(self.reaction_rate)

    def _predict_concentration_slice(self,
                                     initial_concentration: np.ndarray,
                                     time_slice: np.ndarray,
                                     steps_per_step: int) -> tuple[pd.DataFrame, np.ndarray]:
        """
        Predicts the concentration of a singular time slice.
        :param initial_concentration: The initial concentration of the system.
        :param time_slice: The datapoints that must be recorded.
        :param steps_per_step: The number of steps to simulate inbetween each step in the time slice.
        Higher values yield higher accuracy at the cost of computation time.
        :return prediction: pd.Dataframe of the prediction and a np.ndarray of the last prediction step.
        """
        predicted_concentration = np.full((len(time_slice), len(initial_concentration)), np.nan)
        predicted_concentration[0, :] = initial_concentration
        prev_t = time_slice[0]

        # use the given steps
        prediction_step = initial_concentration
        for row, next_spectra_t in enumerate(time_slice[1:]):
            in_between_steps = np.linspace(prev_t, next_spectra_t, steps_per_step + 1)[1:]  # ignore start
            for new_t in in_between_steps:
                prediction_step = calculate_step(
                    reaction_rate=self.reaction_rate,
                    reaction_reactants=self.reaction_reactants,
                    reaction_products=self.reaction_products,
                    concentrations=prediction_step,
                    delta_time=new_t - prev_t, )
                prev_t = new_t

            if any(prediction_step < 0):
                raise ValueError("Negative concentrations were found, increase the steps per step to resolve this. "
                                 f"\n\tThis happened during iteration {row}")
            predicted_concentration[row + 1, :] = prediction_step

        df_result = pd.DataFrame(predicted_concentration, columns=list(self.reference.keys()))
        df_result["time"] = time_slice

        return df_result, prediction_step  # last prediction step

    def predict_concentration(self,
                              experimental_conditions: Experimental_Conditions,
                              steps_per_step: int = 1,
                              ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Predicts the concentration of a system, using the appropriate experimental conditions.
        :param experimental_conditions: Experimental Conditions object.
        :param steps_per_step: The number of steps to simulate inbetween each step in the time slice.
        Higher values yield higher accuracy at the cost of computation time.
        :return (unlabeled prediction, labeled prediction,): Pd.Dataframe of the situation pre-addition of the labeled
         compound, and one of the post-addition situation.
        """
        # reorder the initial concentrations such that they match with the sorting in self.reference
        for compound, initial_concentration in experimental_conditions.initial_concentrations.items():
            self.initial_concentrations[self.reference[compound]] = initial_concentration

        # pre addition
        result_pre_addition, last_prediction = self._predict_concentration_slice(
            initial_concentration=self.initial_concentrations,
            time_slice=experimental_conditions.time[0],
            steps_per_step=steps_per_step
        )

        # dillution step
        diluted_concentrations = last_prediction * experimental_conditions.dilution_factor
        for reactant, concentration in experimental_conditions.labeled_reactant.items():
            diluted_concentrations[self.reference[reactant]] = concentration

        # post addition
        results_post_addition, _ = self._predict_concentration_slice(
            initial_concentration=diluted_concentrations,
            time_slice=experimental_conditions.time[1],
            steps_per_step=steps_per_step
        )
        return result_pre_addition, results_post_addition
