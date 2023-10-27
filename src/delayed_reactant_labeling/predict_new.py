import numpy as np
import pandas as pd
import polars as pl
from numba import njit
from numba.typed import List
from scipy.integrate import solve_ivp


class InvalidPredictionError(Exception):
    """Exception which is raised upon the detection of invalid predictions."""
    pass


@njit
def _calculate_steps_euler(reaction_rate: np.ndarray,
                           reaction_reactants: List[np.ndarray],
                           reaction_products: List[np.ndarray],
                           concentration: np.ndarray,
                           time_slice: np.ndarray,
                           steps_per_step: int):
    """
    Calculates a singular step using the Explicit Euler formula.
    Foreach defined reaction all reactants will be decreased by the 'created amount',
    whereas the products will be increased.
    :param reaction_rate: Each element contains the rate constant values.
    :param reaction_reactants: Each element contains an array of the indices of which chemicals are the reactants.
    :param reaction_products: Each element contains an array of the indices of which chemicals are the products.
    :param concentration: The initial concentrations of each chemical.
    :param time_slice: The points in time that must be examined.
    :param steps_per_step: The number of simulations which are examined, for each point in the time slice.
    :return: The predicted concentrations.
    """
    prediction = np.empty((time_slice.shape[0], concentration.shape[0]))
    prediction[0, :] = concentration

    for time_i in range(time_slice.shape[0] - 1):
        # Step over the total delta t in n steps per step. Discard the intermediate results.
        dt = (time_slice[time_i + 1] - time_slice[time_i]) / steps_per_step
        for _ in range(steps_per_step):
            new_concentration = concentration.copy()
            for reaction_i in range(reaction_rate.shape[0]):
                created_amount = dt * reaction_rate[reaction_i] * np.prod(concentration[reaction_reactants[reaction_i]])
                new_concentration[reaction_reactants[reaction_i]] -= created_amount  # consumed
                new_concentration[reaction_products[reaction_i]] += created_amount  # produced
            concentration = new_concentration

        # update each step
        prediction[time_i + 1, :] = concentration
    return prediction


@njit
def dc_dt(concentrations: np.ndarray,
          reaction_rates: np.ndarray,
          reaction_reactants: List[np.ndarray],
          reaction_products: List[np.ndarray]):
    """
    Calculates the rate of change for each chemical as a function of time
    :param concentrations: The last known concentration of each chemical.
    :param reaction_rates: The rate-constant of each reaction.
    :param reaction_reactants: The indices of the reactants in each reaction.
    :param reaction_products: The indices of the products in each reaction.
    """
    _dc_dt = np.zeros(concentrations.shape)
    for i in range(reaction_rates.shape[0]):
        created_amount = reaction_rates[i] * np.prod(concentrations[reaction_reactants[i]])
        _dc_dt[reaction_reactants[i]] -= created_amount  # consumed
        _dc_dt[reaction_products[i]] += created_amount  # produced
    return _dc_dt


@njit
def _calculate_jac(concentrations: np.ndarray,
                   reaction_rates: np.ndarray,
                   reaction_reactants: list[np.ndarray],
                   reaction_products: list[np.ndarray]):
    """
    Calculates the rate of change for each chemical as a function of time
    :param concentrations: The last known concentration of each chemical.
    :param reaction_rates: The rate-constant of each reaction.
    :param reaction_reactants: The indices of the reactants in each reaction.
    :param reaction_products: The indices of the products in each reaction.
    """
    L = concentrations.shape[0]
    _jac = np.zeros((L, L))

    for r in range(reaction_rates.shape[0]):
        reactants = reaction_reactants[r]
        products = reaction_products[r]
        rate = reaction_rates[r]

        # derivative with respect to chemical j. partial derivatives w.r.t. products are 0.
        for j in range(reactants.shape[0]):
            _reactants_concentrations = concentrations[reactants[reactants != reactants[j]]]

            # chemical i in the reactants
            for i in range(reactants.shape[0]):
                _jac[reactants[i], reactants[j]] -= rate * np.prod(_reactants_concentrations)

            # chemical i in the products
            for i in range(products.shape[0]):
                _jac[products[i], reactants[j]] += rate * np.prod(_reactants_concentrations)
    return _jac


class DRL:
    """Class which enables efficient prediction of changes in concentration in a chemical system.
    Especially useful for Delayed Reactant Labeling (DRL) experiments."""

    def __init__(self,
                 reactions: list[tuple[str, list[str], list[str]]],
                 rate_constants: dict[str, float] | pd.Series,
                 output_order: list[str] = None,
                 verbose: bool = False):
        """
        Initialize the chemical system.
        :param reactions: List of reactions, each reaction is given as a tuple.
        Its first element is a string, which determines which rate constant is applicable to that reaction.
        Its second element is a list containing the identifiers (strings) of each reactant in the reaction.
        The third element contains a list for the reaction products
        :param rate_constants: A dictionary which maps the rate constants to their respective values.
        :param output_order: Defines in which column the concentration of each chemical will be stored.
            By default, it is alphabetical.
        :param verbose: If True, it will print and store information on which reactions are initialized.
        """
        if verbose:
            # Pandas is much more flexible when it comes to storing data. Especially lists in lists.
            df = []
            for k, reactants, products in reactions:
                df.append(pd.Series([k, rate_constants[k], reactants, products],
                                    index=['k', 'k-value', 'reactants', 'products']))
            self.reactions = pd.DataFrame(df)
            print(self.reactions)

        # The rate constants that were inputted will be shown if an error occurs. Allows for easier debugging.
        self.rate_constants_input = pd.Series(rate_constants)

        # link the name of a chemical with an index
        if output_order is None:
            # default to alphabetical order
            chemicals = set()
            for _, reactants, products in reactions:
                for chemical in reactants + products:
                    chemicals.add(chemical)
            output_order = list(sorted(chemicals))

        self.reference = pd.Series(np.arange(len(output_order)), index=output_order)
        self.initial_concentrations = np.zeros((len(self.reference)))  # default is 0 for each chemical

        # construct a list containing the indices of all the reactants and products per reaction
        self.reaction_rate = []  # np array at the end
        self.reaction_reactants = List()  # multiply everything per reaction, and multiply by k
        self.reaction_products = List()  # add

        for k, reactants, products in reactions:
            if rate_constants[k] == 0:
                # the reaction does not create or consume any chemicals, therefore its redundant and can be removed for
                # computational benefits
                continue

            self.reaction_rate.append(rate_constants[k])
            self.reaction_reactants.append(np.array([self.reference[reactant] for reactant in reactants]))
            self.reaction_products.append(np.array([self.reference[product] for product in products]))
        self.reaction_rate = np.array(self.reaction_rate)

    def calculate_step(self, t, y):
        """
        Wrapper around dc_dt() to fix the arguments.
        :param t: Time
        :param y: Concentrations
        """
        return dc_dt(y, self.reaction_rate, self.reaction_reactants, self.reaction_products)

    def calculate_jac(self, t, y):
        return _calculate_jac(y, self.reaction_rate, self.reaction_reactants, self.reaction_products)

    def predict_concentration(self,
                              t_eval_pre,
                              t_eval_post,
                              initial_concentrations: dict[str, float],
                              labeled_concentration: dict[str, float],
                              dilution_factor: float,
                              atol: float = 1e-10,
                              rtol: float = 1e-10):
        """
        Predicts the concentrations during a DRL experiment.
        It utilizes the ODE solver 'scipy.integrate.solve_ivp' with the Radau method.
        :param t_eval_pre: The time steps that must be evaluated and returned before the addition of the labeled compound.
        :param t_eval_post:  The time steps that must be evaluated and returned after the addition of the labeled compound.
        :param initial_concentrations: The initial concentrations of each chemical. Non-zero concentrations are not required.
        :param labeled_concentration: The concentration of the labeled chemical. This concentration is not diluted.
        :param dilution_factor: The factor (<1) by which the prediction will be 'diluted' when the labeled chemical is added.
        :param atol: The absolute tolerances for the ODE solver.
        :param rtol: The relative tolerances for the ODE solver.
        """
        # modify the stored initial concentration to match with input.
        for chemical, initial_concentration in initial_concentrations.items():
            self.initial_concentrations[self.reference[chemical]] = initial_concentration

        jac_sparsity = None # self.calculate_jac(None, np.ones(self.initial_concentrations.shape))

        result_pre = solve_ivp(self.calculate_step,
                               t_span=[t_eval_pre[0], t_eval_pre[-1]],
                               t_eval=t_eval_pre,
                               y0=self.initial_concentrations,
                               jac=self.calculate_jac,
                               method='Radau',
                               jac_sparsity=jac_sparsity,
                               atol=atol,
                               rtol=rtol)
        df_result_pre = pl.DataFrame(result_pre.y, list(self.reference.keys()))
        df_result_pre = df_result_pre.with_columns(pl.Series(name='time', values=result_pre.t))

        # dilution step
        diluted_concentrations = result_pre.y[:, -1] * dilution_factor  # result.y is transposed
        for chemical, concentration in labeled_concentration.items():
            diluted_concentrations[self.reference[chemical]] = concentration

        # post addition
        result_post = solve_ivp(self.calculate_step,
                                t_span=[t_eval_post[0], t_eval_post[-1]],
                                t_eval=t_eval_post,
                                y0=diluted_concentrations,
                                method='Radau',
                                jac=self.calculate_jac,
                                jac_sparsity=jac_sparsity,
                                atol=atol,
                                rtol=rtol)
        df_result_post = pl.DataFrame(result_post.y, list(self.reference.keys()))
        df_result_post = df_result_post.with_columns(pl.Series(name='time', values=result_post.t))

        # validate the results
        if result_post.y.min() < -max([atol, rtol]):  # errors up to the given tolerance are allowed.
            raise InvalidPredictionError(
                f"Negative concentrations (min: {result_post.y.min():6e}) were detected. "
                f"The applied rate constants are:\n {self.rate_constants_input.to_json()}")
        if np.isnan(df_result_post.tail(1)).any():
            raise InvalidPredictionError(
                f"NaN values (count: {np.isnan(df_result_post.to_numpy).sum()}) were detected. "
                f"The applied rate constants are:\n {self.rate_constants_input.to_json()}")

        return df_result_pre, df_result_post

    def _predict_slice_Euler(self,
                             initial_concentration: np.ndarray,
                             time_slice: np.ndarray,
                             steps_per_step: int) -> tuple[pl.DataFrame, np.ndarray]:
        """
        Predicts the concentration of a singular time slice.
        :param initial_concentration: The initial concentration of the system.
        :param time_slice: The datapoints that must be recorded.
        :param steps_per_step: The number of steps to simulate inbetween each step in the time slice.
        Higher values yield higher accuracy at the cost of computation time.
        :return prediction: pd.Dataframe of the prediction and a np.ndarray of the last prediction step.
        """
        # calculate all steps of the time slice
        predicted_concentration = _calculate_steps_euler(
            reaction_rate=self.reaction_rate,
            reaction_reactants=self.reaction_reactants,
            reaction_products=self.reaction_products,
            concentration=initial_concentration,
            time_slice=time_slice,
            steps_per_step=steps_per_step)

        # do some formatting
        df_result = pl.DataFrame(predicted_concentration, list(self.reference.keys()))
        df_result = df_result.with_columns(pl.Series(name='time', values=time_slice))
        return df_result, predicted_concentration[-1, :]

    def predict_concentration_Euler(self,
                                    t_eval_pre,
                                    t_eval_post,
                                    initial_concentrations: dict[str, float],
                                    labeled_concentration: dict[str, float],
                                    dilution_factor: float,
                                    steps_per_step=1):
        """
        Predicts the concentrations during a DRL experiment.
        :param t_eval_pre: The time steps that must be evaluated and returned before the addition of the labeled compound.
        :param t_eval_post:  The time steps that must be evaluated and returned after the addition of the labeled compound.
        :param initial_concentrations: The initial concentrations of each chemical. Non-zero concentrations are not required.
        :param labeled_concentration: The concentration of the labeled chemical. This concentration is not diluted.
        :param dilution_factor: The factor (<1) by which the prediction will be 'diluted' when the labeled chemical is added.
        :param steps_per_step: The number of steps to simulate inbetween each step in the time slice.
        """
        # modify the stored initial concentration to match with input.
        for chemical, initial_concentration in initial_concentrations.items():
            self.initial_concentrations[self.reference[chemical]] = initial_concentration

        # pre addition
        result_pre_addition, last_prediction = self._predict_slice_Euler(
            initial_concentration=self.initial_concentrations,
            time_slice=t_eval_pre,
            steps_per_step=steps_per_step
        )

        # dillution step
        diluted_concentrations = last_prediction * dilution_factor
        for reactant, concentration in labeled_concentration.items():
            diluted_concentrations[self.reference[reactant]] = concentration

        # post addition
        results_post_addition, _ = self._predict_slice_Euler(
            initial_concentration=diluted_concentrations,
            time_slice=t_eval_post,
            steps_per_step=steps_per_step
        )

        # validate the results
        if results_post_addition.to_numpy().min() < 0:  # some leniency is required
            raise InvalidPredictionError(
                "Negative concentrations were detected, perhaps this was caused by a large dt.\n"
                "Consider increasing the steps_per_step. The applied rate constants are:\n"
                f"{self.rate_constants_input.to_json()}")
        if np.isnan(results_post_addition.tail(1)).any():
            raise InvalidPredictionError(
                "NaN values were detected in the prediction, perhaps this was caused by a large dt.\n"
                "Consider increasing the steps_per_step. The applied rate constants are:\n"
                f"\n{self.rate_constants_input.to_json()}"
            )

        return result_pre_addition, results_post_addition
