import numpy as np
import pandas as pd
import polars as pl
from numba import njit
from numba.typed import List
from scipy.integrate import solve_ivp


raise NotImplementedError
# this doesnt work for idk what reason
# TODO compare jac with finite diff aproximation


class InvalidPredictionError(Exception):
    """Exception which is raised upon the detection of invalid predictions."""
    pass


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


def jac(concentrations: np.ndarray,
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

    for r in range(len(reaction_rates)):
        reactants = reaction_reactants[r]
        products = reaction_products[r]

        rate = reaction_rates[r]

        chemicals = np.append(reactants, products)
        for chemical_i in chemicals:
            for chemical_j in chemicals:
                if any(chemical_i == reactants):
                    sign = -1  # consumed by this sub reaction
                else:
                    sign = +1  # produced by this sub reaction

                eq = chemical_j == reactants
                if any(eq):
                    _reactants = reactants[~eq]  # only use all non-chemical_j reactants; this is a differentiation step
                else:
                    _reactants = reactants

                _jac[chemical_i, chemical_j] += rate * np.prod(concentrations[_reactants])
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

    def jac_wrap(self, t, y):
        return jac(y, self.reaction_rate, self.reaction_reactants, self.reaction_products)

    def predict_concentration(self,
                              t_eval_pre,
                              t_eval_post,
                              initial_concentrations: dict[str, float],
                              labeled_concentration: dict[str, float],
                              dilution_factor: float,
                              ivp_method: str = "Radau",
                              **solve_ivp_kw):
        """
        Predicts the concentrations during a DRL experiment.
        :param t_eval_pre: The time steps that must be evaluated and returned before the addition of the labeled compound.
        :param t_eval_post:  The time steps that must be evaluated and returned after the addition of the labeled compound.
        :param initial_concentrations: The initial concentrations of each chemical. Non-zero concentrations are not required.
        :param labeled_concentration: The concentration of the labeled chemical. This concentration is not diluted.
        :param dilution_factor: The factor (<1) by which the prediction will be 'diluted' when the labeled chemical is added.
        :param ivp_method: The method used by scipy.integrate.solve_ivp. LSODA (default) has automatic stiffness detection.
        :param solve_ivp_kw: All keyword arguments will be passed to scipy.integrate.solve_ivp.
        """
        # modify the stored initial concentration to match with input.
        for chemical, initial_concentration in initial_concentrations.items():
            self.initial_concentrations[self.reference[chemical]] = initial_concentration

        # pre addition
        # c = self.initial_concentrations
        # dt = 1e-12
        # for _ in range(10):
        #     dc = self.calculate_step(None, c)
        #     c = c + dt*dc
        # self.initial_concentrations = c

        result_pre = solve_ivp(self.calculate_step,
                               t_span=t_eval_pre[0, -1],
                               t_eval=t_eval_pre,
                               y0=self.initial_concentrations,
                               jac=self.jac_wrap,
                               method=ivp_method,
                               **solve_ivp_kw)
        df_result_pre = pl.DataFrame(result_pre.y, list(self.reference.keys()))
        df_result_pre = df_result_pre.with_columns(pl.Series(name='time', values=result_pre.t))

        # dilution step
        diluted_concentrations = result_pre.y[:, -1] * dilution_factor  # result.y is transposed
        for chemical, concentration in labeled_concentration.items():
            diluted_concentrations[self.reference[chemical]] = concentration

        # post addition
        result_post = solve_ivp(self.calculate_step,
                                t_span=t_eval_post[0, -1],
                                t_eval=t_eval_post,
                                y0=diluted_concentrations,
                                method=ivp_method,
                                jac=self.jac_wrap,
                                **solve_ivp_kw)
        df_result_post = pl.DataFrame(result_post.y, list(self.reference.keys()))
        df_result_post = df_result_post.with_columns(pl.Series(name='time', values=result_post.t))

        # validate the results
        # if result_post.y.min() < 0:
        #     raise InvalidPredictionError(
        #         "Negative concentrations were detected, perhaps this was caused by a large dt.\n"
        #         "Consider increasing the steps_per_step. The applied rate constants are:\n"
        #         f"{self.rate_constants_input.to_json()}")
        # if np.isnan(df_result_post.tail(1)).any():
        #     raise InvalidPredictionError(
        #         "NaN values were detected in the prediction, perhaps this was caused by a large dt.\n"
        #         "Consider increasing the steps_per_step. The applied rate constants are:\n"
        #         f"\n{self.rate_constants_input.to_json()}"
        #     )

        return df_result_pre, df_result_post

