import os
import warnings

import numpy as np
import pandas as pd  # used for storage of the data, its series objects are much more powerful.
import polars as pl  # used for more efficient calculations in the dataframes.

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, Callable
from copy import deepcopy

from joblib import Parallel, delayed
from scipy.optimize import minimize
from scipy.stats import loguniform
from tqdm import tqdm

from delayed_reactant_labeling.predict import InvalidPredictionError


class JSON_log:
    def __init__(self, path, mode="new"):
        self._path = path
        exists = os.path.isfile(path)

        if mode == "new":
            # create a new file
            if exists:
                raise FileExistsError(f"{path} already exists. To replace it use mode='replace'")
            with open(self._path, "w") as _:
                pass

        elif mode == "append":
            # append to the file
            if not exists:
                raise ValueError(f"{path} does not exist. Use mode='new' to create it.")

        elif mode == "replace":
            # replace the file
            with open(self._path, "w") as _:
                pass

    def log(self, data: pd.Series):
        data["datetime"] = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        with open(self._path, "a") as f:
            f.write(data.to_json() + "\n")


class OptimizerProgress:
    def __init__(self, path: str):
        # read the meta data
        self.metadata = pd.read_json(f"{path}/settings_info.json", lines=True).iloc[0, :]
        self.x_description = np.array(self.metadata["x_description"])

        # read the optimization log
        df = pd.read_json(f"{path}/optimization_log.json", lines=True)
        self.n_dimensions = len(self.x_description)
        self.n_iterations = len(df)

        self.all_X: pd.DataFrame = pd.DataFrame(list(df.loc[:, "x"]), columns=self.x_description)
        self.all_errors: pd.Series = df["error"]
        self.all_times: pd.Series = df["datetime"]

        simplex = np.full((self.n_dimensions + 1, self.n_dimensions), np.nan)
        sorted_errors = self.all_errors.sort_values(ascending=True)
        for n, index in enumerate(sorted_errors[:self.n_dimensions + 1].keys()):
            simplex[n, :] = self.all_X.iloc[index, :].to_numpy()  # underscored variant so that no copying is required

        best_iteration_index = sorted_errors.index[0]

        self.best_X: pd.Series = pd.Series(self.all_X.loc[best_iteration_index, :], index=self.x_description)
        self.best_error: float = self.all_errors[best_iteration_index]


class RateConstantOptimizerTemplate(ABC):
    def __init__(self,
                 raw_weights: dict[str, float],
                 experimental: pl.DataFrame,
                 metric: Callable[[np.ndarray, np.ndarray], float]) -> None:
        """
        Initializes the Rate constant optimizer class.
        :param raw_weights: Dictionary containing str patterns as keys and the weight as values.
        The str patterns will be searched for in the keys of the results from the 'calculate_curves' function.
        The given weight will lower corresponding errors.
        :param experimental: Polars dataframe containing the experimental data.
        :param metric: The error metric which should be used. It must implement y_true and y_pred as its arguments.
        """
        self.raw_weights = raw_weights
        self.weights = None

        # initialize all curves for the experimental (true) values.
        self.experimental_curves = self.calculate_curves(experimental)
        self.metric = metric

        # check if any of the curves are potentially problematic
        nan_warnings = []  #
        for curve_description, curve in self.experimental_curves.items():
            if curve.is_nan().any():
                nan_warnings.append(f"Experimental data curve for {curve_description} contains {curve.is_nan().sum()} NaN values.")
        if nan_warnings:
            warnings.warn("\n".join(nan_warnings))

    @staticmethod
    @abstractmethod
    def create_prediction(x: np.ndarray, x_description: list[str]) -> pl.DataFrame:
        """
        Create a prediction of the system, given a set of parameters.
        :param x: Contains all parameters, which are to be optimized.
        Definitely includes are rate constants.
        :param x_description: Contains a description of each parameter.
        :return: Predicted values of the concentration for each chemical, as a function of time.
        """
        pass

    @staticmethod
    @abstractmethod
    def calculate_curves(data: pl.DataFrame) -> dict[str, pl.Series]:
        """
        Calculate the curves corresponding to the data (either experimental or predicted).
        The experimental curves will only be calculated once and are stored for subsequent use.
        Internally, the experimental and predicted curves will be compared against each other to determine the error.
        :param data: The data from which the curves should be calculated
        :return: dict containing a description of each curve, and the corresponding curve.
        """

    def calculate_error_functions(self, prediction: pl.DataFrame) -> pd.Series:
        """
        Calculate the error caused by each error function.
        The input is of the format of polars due to its computational efficiency.
        It returns a Pandas Series as those are more powerful to work with.
        :param prediction: Polars dataframe containing the predicted concentrations.
        :return: Pandas.Series containing the unweighted errors of each error function.
        """
        curves_predicted = self.calculate_curves(prediction)
        error = {}
        nan_warnings = []
        for curve_description, curve_prediction in curves_predicted.items():
            # noinspection PyArgumentList
            error[curve_description] = self.metric(
                y_true=self.experimental_curves[curve_description],
                y_pred=curve_prediction)
            if np.isnan(error[curve_description]):
                nan_warnings.append(f"The error function for {curve_description} is NaN.")
        if nan_warnings:
            raise ValueError("\n".join(nan_warnings))
        return pd.Series(error)

    def weigh_errors(self, errors: pd.Series, ) -> pd.Series:
        """
        weighs the errors
        :param errors: unweighted errors
        :return: weighed errors
        """
        # assert isinstance(errors, pl.Series)
        if self.weights is None:
            weights = np.ones(errors.shape)
            for description, weight in self.raw_weights.items():
                index = errors.index.str.contains(description)
                if len(index) == 0:
                    raise ValueError(f"no matches were found for {description}")
                weights[index] = weights[index] * weight
            self.weights = weights

        return errors * self.weights

    def calculate_total_error(self, errors: pd.Series) -> float:
        """
        weighs and sums the errors.
        :param errors: unweighted errors
        :return: weighed total error
        """
        return self.weigh_errors(errors).sum(skipna=False)

    def optimize(self,
                 x0: np.ndarray,
                 x_description: list[str],
                 x_bounds: list[tuple[float, float]],
                 path: str,
                 metadata: Optional[dict] = None,
                 maxiter: float = 50000,
                 resume_from_simplex=None,
                 pbar_show=True,
                 _overwrite_log=False,
                 **tqdm_kwargs
                 ) -> None:
        """
        Optimizes the system, utilizing a nelder-mead algorithm.
        :param x0: Parameters which are to be optimized. Always contain the rate constants.
        :param x_description: Description of each parameter.
        :param x_bounds: A list containing tuples, which in turn contain the lower and upper bound for each parameter.
        :param path: Where the solution should be stored.
        :param metadata: The metadata that should be saved alongside the solution.
        :param maxiter: The maximum number of iterations.
        :param resume_from_simplex: When a simplex is given, the solution starts here.
        :param pbar_show: If True, shows a progress bar.
        :param tqdm_kwargs: Keyword arguments for the progress bar (tqdm).
        :param _overwrite_log: If True, the logs will be overwritten.
            Should only be used in test scripts to avoid accidental loss of data.
        """
        log_mode = "new" if not _overwrite_log else "replace"

        # enable logging of all information retrieved from the system
        log_path = f"{path}/optimization_log.json"
        if resume_from_simplex is None:  # new optimization progres
            logger = JSON_log(log_path, mode=log_mode)
            metadata_extended = {
                "raw_weights": self.raw_weights,
                "x0": x0,
                "x_description": x_description,
                "bounds": x_bounds,
                "maxiter": maxiter
            }
            if metadata is not None:
                # overwrites the default meta data values
                for key, value in metadata.items():
                    metadata_extended[key] = value
            meta_data_log = JSON_log(f"{path}/settings_info.json", mode=log_mode)
            meta_data_log.log(pd.Series(metadata_extended))
        else:
            logger = JSON_log(log_path, mode="append")

        def optimization_step(x):
            """The function is given a set of parameters by the Nelder-Mead algorithm.
            Proceeds to calculate the corresponding prediction and its total error.
            The results are stored in a log before the error is returned to the optimizer."""
            prediction = self.create_prediction(x, x_description)
            errors = self.calculate_error_functions(prediction)
            total_error = self.calculate_total_error(errors)

            logger.log(pd.Series([x, total_error], index=["x", "error"]))
            return total_error

        try:
            if pbar_show:
                def update_tqdm(_):
                    """update the progress bar"""
                    pbar.update(1)

                with tqdm(total=maxiter, miniters=25, **tqdm_kwargs) as pbar:
                    # the minimization process is stored within the log, containing all x's and errors.
                    minimize(fun=optimization_step,
                             x0=x0,
                             method="Nelder-Mead",
                             bounds=x_bounds,
                             callback=update_tqdm,
                             options={"maxiter": maxiter, "disp": True, "adaptive": True, "return_all": False,
                                      "initial_simplex": resume_from_simplex})
            else:
                # the minimization process is stored within the log, containing all x's and errors.
                minimize(fun=optimization_step,
                         x0=x0,
                         method="Nelder-Mead",
                         bounds=x_bounds,
                         options={"maxiter": maxiter, "disp": True, "adaptive": True, "return_all": False,
                                  "initial_simplex": resume_from_simplex})
        except Exception as e:
            logger.log(pd.Series({'MAE': np.nan, 'exception': e}))
            raise e

    def optimize_multiple(self,
                          path: str,
                          n_runs: int,
                          x_description: list[str],
                          x_bounds: list[tuple[float, float]],
                          x0_bounds: Optional[list[tuple[float, float]]] = None,
                          x0_min: float = 1e-6,
                          n_jobs: int = 1,
                          backend: str = "loky",
                          batch_size: int = "auto",
                          **optimize_kwargs):
        """
        Optimizes the system, utilizing a nelder-mead algorithm, for a given number of runs. Each run has random
        starting positions for each parameter, which is distributed according to a loguniform distribution. The bounds
        of the starting position (x0_bounds) can be separately controlled from the bounds the system is allowed to
        explore (x_bounds). If the given path already has an existing directory called 'optimization_multiple_guess',
        the optimization will be resumed from that point onwards.
        :param path: Where the solution should be stored.
        :param n_runs: The number of runs which are to be computed.
        :param x_description: Description of each parameter.
        :param x_bounds: A list containing tuples, containing the lower and upper boundaries for each parameter.
        :param x0_bounds: A list containing tuples, containing the lower and upper boundaries for the starting value of
            each parameter. By default, it is identical to the x_bounds. Lower bounds smaller than x0_min are set to x0_min.
        :param x0_min: The minimum value the lower bound of x0_bounds can take. Any values lower than it, are set to
            x0_min.
        :param n_jobs: The number of processes which should be used, if -1, all available cores are used.
        :param backend: The backend used by joblib.
        :param batch_size: The size of each batch used by joblib.
        :param optimize_kwargs: The key word arguments that will be passed to self.optimize.
        """
        try:
            os.mkdir(path)
            start_seed = 0
        except FileExistsError:
            start_seed = len(os.listdir(path))
            warnings.warn("Cannot create a directory when that directory already exists. "
                          f"Appending results instead starting with seed {start_seed}")

        if x0_bounds is None:
            x0_bounds = deepcopy(x_bounds)

        x0_bounds = [(lb, ub,) if lb > x0_min else (x0_min, ub) for lb, ub in x0_bounds]

        Parallel(n_jobs=n_jobs, verbose=100, backend=backend, batch_size=batch_size)(
            delayed(self._mp_work_list)(
                seed=seed,
                x_description=x_description,
                x_bounds=x_bounds,
                x0_bounds=x0_bounds,
                path=path,
                optimize_kwargs=optimize_kwargs
            )
            for seed in range(start_seed, start_seed + n_runs)
        )

    def _mp_work_list(self, seed, x_description, x_bounds, x0_bounds, path, optimize_kwargs):
        rv = loguniform
        rv.random_state = seed
        x0 = np.array([rv.rvs(lb, ub) if ub > 0 else 0 for lb, ub in x0_bounds])
        path = f'{path}/guess_{seed}/'
        os.mkdir(path)

        try:
            self.optimize(
                x0=x0,
                x_description=x_description,
                x_bounds=x_bounds,
                path=path,
                pbar_show=False,
                **optimize_kwargs
            )
        except InvalidPredictionError as e:
            warnings.warn(f"Invalid prediction was found for seed {seed}: {e}")
            pass  # results are stored incase an error occurred due to self.optimize.

    @staticmethod
    def load_optimization_progress(path: str) -> OptimizerProgress:
        """
        Loads in the data from the log files.
        :param path: Folder in which the optimization_log.json and settings_info.json files can be found.
        :return: OptimizerProgress instance which contains all information that was logged.
        """
        return OptimizerProgress(path)
