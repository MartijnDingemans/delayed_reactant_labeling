from __future__ import annotations

import json
import os
import pathlib
import warnings
from abc import ABC, abstractmethod
from datetime import datetime
from math import log
from random import random
from typing import Optional, Callable
import numpy as np
import pandas as pd  # used for storage of the data, its series objects are much more powerful.
from delayed_reactant_labeling.predict import InvalidPredictionError
from delayed_reactant_labeling.optimize_nelder_mead import minimize_neldermead
from joblib import Parallel, delayed
from scipy.optimize import Bounds
from tqdm import tqdm


class JSON_log:
    def __init__(self, path, mode="new"):
        self._path = path
        exists = os.path.isfile(path)

        if mode == "new":
            # create a new file
            if exists:
                raise FileExistsError(f"{path} already exists.")
            with open(self._path, "w") as _:
                pass

        elif mode == "append":
            # append to the file
            if not exists:
                raise ValueError(f"{path} does not exist.")

        elif mode == "replace":
            # replace the file
            with open(self._path, "w") as _:
                pass

    def log(self, data: pd.Series):
        data["datetime"] = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        with open(self._path, "a") as f:
            f.write(data.to_json() + "\n")


class OptimizedModel:
    """Formatted data structure of an optimized model.

    Parameters
    ----------
    path
        The path to the folder where the optimization progress has been stored.

    Attributes
    ----------
    metadata : dict[str, any]
        The stored metadata.
    x_description : list[str]
        The description of each parameter.
    n_dimensions : int
        The number of parameters.
    n_function_evaluations : int
        The number of computed iterations.
    all_x : pd.DataFrame
        The applied parameters for each iteration.
    all_errors : pd.Series
        The error for each iteration.
    all_times : pd.Series
        The timestamp at which the computation of each iteration was finished.
    optimal_x : pd.Series
        The parameters for the iteration with the lowest error.
    optimal_error : float
        The error corresponding to the iteration with the lowest error.
    simplex : np.ndarray
        An array of size [N + 1, N] corresponding to the N parameters for each of the N + 1 best iterations
    """
    def __init__(self, path: str | pathlib.Path):
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)

        if not path.is_dir():
            raise ValueError(f'The given path was not a directory!\n{path}')

        # read the meta data
        self.metadata: dict[str, any] = pd.read_json(path.joinpath("settings_info.json"), lines=True).iloc[0, :]
        self.x_description = pd.Index(self.metadata["x_description"])

        # read the optimization log
        log_path = path.joinpath("optimization_log.json")
        try:
            df = pd.read_json(log_path, lines=True)
        except ValueError:
            with open(log_path) as f:
                lines = f.readlines()
            df = []
            n_failed = 0
            for line in lines:
                try:
                    df.append(json.loads(line))
                except ValueError:
                    n_failed += 1
            df = pd.DataFrame(df)
            warnings.warn(f'failed to read {n_failed} lines in {log_path}')

        self.n_dimensions = len(self.x_description)
        self.n_function_evaluations = len(df)

        self.all_x: pd.DataFrame = pd.DataFrame(list(df.loc[:, "x"]), columns=self.x_description)
        self.all_errors: pd.Series = df["error"]
        self.all_times: pd.Series = df["datetime"]

        simplex = np.full((self.n_dimensions + 1, self.n_dimensions), np.nan)
        sorted_errors = self.all_errors.sort_values(ascending=True)
        for n, index in enumerate(sorted_errors[:self.n_dimensions + 1].keys()):
            simplex[n, :] = self.all_x.iloc[index, :].to_numpy()
        self.simplex = simplex

        # noinspection PyUnresolvedReferences
        best_iteration_index = sorted_errors.index[0]

        self.initial_x: pd.Series = pd.Series(self.all_x.iloc[0, :], index=self.x_description)
        self.optimal_x: pd.Series = pd.Series(self.all_x.iloc[best_iteration_index, :], index=self.x_description)
        self.optimal_error: float = self.all_errors[best_iteration_index]


class OptimizedMultipleModels:
    """Formatted data structure for multiple optimized models.
    The data will be formatted such that the best run starts at index 0,
    and the error ascends with each index number.

    Parameters
    ----------
    path
        The path to the directory, which contains subdirectories.
        Each subdirectory in turn should contain its optimization progress.

    Attributes
    ----------
    x_description : list[str]
        The description of each parameter.
    all_initial_x : pd.DataFrame
        The initial parameters, x, in each model.
    all_optimal_x : pd.DataFrame
        The optimal parameters, x, in each model.
    all_optimal_error : np.ndarray
        The error when using the optimal parameters in each model.
    best : OptimizedModel
        The model with the lowest error.
    """
    def __init__(self, path: str | pathlib.Path):
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)

        # Store general information about each model.
        # This does not contain the model.all_X attribute as those are quite large, and often not used.
        index = []
        initial_x = []
        optimal_x = []
        found_error = []
        self._model_paths = []
        self.x_description = None

        for model_path in tqdm(path.iterdir()):
            if not model_path.is_dir():
                continue
            try:
                model = OptimizedModel(model_path)
            except Exception as e:
                warnings.warn(f'The model at path {model_path} gave the error:\n{repr(e)}')
                continue

            self._model_paths.append(model_path)
            index.append(model_path.parts[-1])  # guess_42
            initial_x.append(model.initial_x)
            optimal_x.append(model.optimal_x)
            found_error.append(model.optimal_error)

            if self.x_description is None:
                self.x_description = model.x_description

        found_error = np.array(found_error)
        best_model_index = np.argsort(found_error)

        self.all_initial_x: pd.DataFrame = pd.DataFrame(initial_x, index=index, columns=self.x_description).iloc[best_model_index, :]
        self.all_optimal_x: pd.DataFrame = pd.DataFrame(optimal_x, index=index, columns=self.x_description).iloc[best_model_index, :]
        self.all_optimal_error: np.ndarray = found_error[best_model_index]

        self._best_model_index = best_model_index
        self.best = self.get_model(0)

    def get_model(self, n: int) -> OptimizedModel:
        """Returns the n-th model, when sorted by the error (the best model at index 0).

        Args
        ----
        n
            The index of the model that is to be retrieved.
        """
        return OptimizedModel(path=self._model_paths[self._best_model_index[n]])


class RateConstantOptimizerTemplate(ABC):
    """Enables easy optimization of a model, which must be semi-implemented by the user.

    Note
    ----
    The user must implement the following abstract functions:

    1.  :meth:`create_prediction`
    2.  :meth:`calculate_curves`

    Parameters
    ------
    experimental
        The experimental data
    metric
        An error metric which takes as input two np.ndarrays for the keywords ``y_pred`` and ``y_true``
        and returns a float. Lower values should indicate a better prediction.
    raw_weights
        A dictionary containing patterns and weight. Each pattern will be searched for in the errors.
        Upon a match the error yielded will be multiplied by its respective weight. If an error is matched with multiple
        patterns, the weight will be decreased in a multiplicative manner. If None (default), no weights will be applied.

    Attributes
    ----------
    weights : Optional[np.ndarray]
        The final weight per error type.
    """
    def __init__(self,
                 experimental: pd.DataFrame,
                 metric: Callable[[np.ndarray, np.ndarray], float],
                 raw_weights: Optional[dict[str, float]] = None,) -> None:

        if raw_weights is None:
            raw_weights = {}

        self.raw_weights = raw_weights
        self.weights: Optional[np.ndarray] = None

        # initialize all curves for the experimental (true) values.
        self.experimental_curves = self.calculate_curves(experimental)
        self.metric = metric

        # check if any of the curves are potentially problematic
        nan_warnings = []  #
        for curve_description, curve in self.experimental_curves.items():
            if np.isnan(curve).any():
                nan_warnings.append(
                    f"Experimental data curve for {curve_description} contains {np.isnan(curve).sum()} NaN values.")

        if nan_warnings:
            warnings.warn("\n".join(nan_warnings))

    @staticmethod
    @abstractmethod
    def create_prediction(x: np.ndarray, x_description: list[str]) -> pd.DataFrame:
        """Creates a prediction of the system, given a set of parameters.
        For DRL experiments :meth:`DRL.predict_concentration <predict.DRL.predict_concentration>`
        does most of the required calculations.

        Args
        ----
        x
            Contains all parameters, which are to be optimized.
        x_description
            The description of each parameter.

        Returns
        -------
        pd.DataFrame
            A DataFrame that contains the predicted concentrations as a function of time.
        """
        pass

    @staticmethod
    @abstractmethod
    def calculate_curves(data: pd.DataFrame) -> dict[str, np.ndarray]:
        """Calculates the curves corresponding to the data (either experimental or predicted).
        The experimental curves will only be calculated upon initialization and are stored for subsequent use.

        Args
        ----
        data
            The data from which the curves should be calculated. Either experimental or predicted by :meth:`create_prediction`.

        Returns
        -------
        dict[str, np.ndarray]
            A dictionary containing a description of each curve, and the corresponding curve.
        """

    def calculate_errors(self, prediction: pd.DataFrame) -> pd.Series:
        """Calculates the (unweighted) error caused by each error function.
        This is done by calculating the curves corresponding to the prediction,
        and comparing them with the experimental curves using the metric.

        Args
        ----
        prediction
            The predicted concentrations as a function of time

        Returns
        -------
        pd.Series
            The unweighted errors of each error function.

        Raises
        ------
        ValueError
            When the metric returns a nan value, a ValueError will be raised. The error message will detail information
            on which curves caused the metric to return nan.
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
        """Weighs the errors.

        Args
        ----
        errors
            The unweighted errors

        Returns
        -------
        pd.Series
            The weighted errors
        """
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
        """Weighs and sums the errors.

        Args
        ----
        errors
            The unweighted errors

        Returns
        -------
        float
            The total error in the model.
        """
        return self.weigh_errors(errors).sum(skipna=False)

    def optimize(self,
                 x0: np.ndarray,
                 x_description: list[str],
                 x_bounds: Bounds,
                 path: str | pathlib.Path,
                 metadata: Optional[dict] = None,
                 maxiter: float = 50000,
                 resume_from_simplex: np.ndarray = None,
                 show_pbar: bool = True,
                 _overwrite_log: bool = False,
                 ) -> None:
        """Optimizes the system, utilizing a nelder-mead algorithm.

        Args
        ----
        x0
            Parameters which are to be optimized. Always contains the rate constants.
        x_description
            Description of each parameter.
        x_bounds
            The scipy.optimize.bounds of each parameter.
        path
            The path to the folder where the optimization progress should be stored.
            The entire path to the folder will be created if it did not exist.
        metadata
            The metadata that should be saved alongside the solution.
            This data will be stored in the settings_info.json file.
        maxiter
            The maximum number of iterations.
        resume_from_simplex
            When a simplex is given of size [N+1, N] where N is the number of parameters, the solution starts here.
            This can be used to resume the optimization process.
        show_pbar
            If True, shows a progress bar.
        _overwrite_log
            If True, the logs will be overwritten.
            Should only be used in test scripts to avoid accidental loss of data.

        Returns
        -------
        None
            All relevant metadata and progress on each iteration will be stored in path. It can be loaded and analyzed
            either by using the :class:`visualize.VisualizeModel`, or :meth:`load_optimization_progress <optimize.RateConstantOptimizerTemplate.load_optimization_progress>` which returns
            an :class:`optimize.OptimizedModel` instance.
        """
        log_mode = "new" if not _overwrite_log else "replace"

        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)
        path.mkdir(exist_ok=True, parents=True)

        # enable logging of all information retrieved from the system
        log_path = path.joinpath(f'optimization_log.json')
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
            meta_data_log = JSON_log(path.joinpath("settings_info.json"), mode=log_mode)
            meta_data_log.log(pd.Series(metadata_extended))
        else:
            logger = JSON_log(log_path, mode="append")

        def optimization_step(x: np.ndarray):
            """The function is given a set of parameters by the Nelder-Mead algorithm.
            Proceeds to calculate the corresponding prediction and its total error.
            The results are stored in a log before the error is returned to the optimizer."""
            prediction = self.create_prediction(x, x_description)
            errors = self.calculate_errors(prediction)
            total_error = self.calculate_total_error(errors)

            logger.log(pd.Series([x, total_error], index=["x", "error"]))
            return total_error

        try:
            if show_pbar:
                def update_tqdm(_):
                    """update the progress bar"""
                    pbar.update(1)

                with tqdm(total=maxiter, miniters=25) as pbar:
                    # the minimization process is stored within the log, containing all x's and errors.
                    minimize_neldermead(
                        func=optimization_step,
                        x0=x0,
                        bounds=x_bounds,
                        callback=update_tqdm,
                        maxiter=maxiter,
                        adaptive=True,
                        initial_simplex=resume_from_simplex)
            else:
                # the minimization process is stored within the log, containing all x's and errors.
                minimize_neldermead(
                    func=optimization_step,
                    x0=x0,
                    bounds=x_bounds,
                    maxiter=maxiter,
                    adaptive=True,
                    initial_simplex=resume_from_simplex)
        except Exception as e:
            logger.log(pd.Series({'MAE': np.nan, 'exception': e}))
            raise e

    def optimize_multiple(self,
                          path: str | pathlib.Path,
                          n_runs: int,
                          x_description: list[str],
                          x_bounds: Bounds,
                          x0_bounds: Optional[Bounds] = None,
                          x0_min: float = 1e-6,
                          n_jobs: int = 1,
                          backend: str = "loky",
                          **optimize_kwargs) -> None:
        """Optimizes the system, utilizing a nelder-mead algorithm, for a given number of runs.
        Each run has random starting positions for each parameter, which is distributed according to a loguniform
        distribution. This choice of distribution allows the model to sample each parameter equally a logarithmic scale,
        for example k1 will be sampled just as often between 10e-5 and 10e-4 as between 0.1 and 1, allowing a wide
        range of values to be examined. The bounds of the starting position (x0_bounds) can be separately controlled
        from the bounds the system is allowed to explore (x_bounds). If the given path already has an existing directory
        called 'optimization_multiple_guess', the optimization will be resumed from that point onwards.

        Args
        ----
        path
            The path to the folder where the optimization progress should be stored.
        n_runs
            The number of runs which are to be computed.
        x_description
            Description of each parameter.
        x_bounds
            The scipy.optimize.bounds of each parameter.
        x0_bounds
            The scipy.optimize.bounds for the starting value of each parameter.
            By default, it is identical to the x_bounds. Lower bounds smaller than x0_min are set to x0_min.
            When the upper bound is 0, the corresponding x0 will also be set to 0. This disables the reaction.
        x0_min
            The minimum value the lower bound of x0_bounds can take. Any values lower than it is set to x0_min.
        n_jobs
            The number of processes which should be used, allowing negative indexing, i.e. n_jobs=-1 would use up all
            available cores, n_jobs=-2 would use all available cores except one, etc.
        backend
            The backend that is used by Joblib. Loky (default) works on all platforms.
        **optimize_kwargs
            The key word arguments that will be passed to self.optimize.

        Returns
        -------
        None
            All data of run n will be stored at 'path/guess_n/'.
        """
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)

        try:
            path.mkdir(exist_ok=False)  # actually, it is okay, but a warning should be given!
            start_seed = 0
        except FileExistsError:
            start_seed = len(os.listdir(path))
            warnings.warn(f"Cannot create a directory ({path}) when that directory already exists. "
                          f"Appending results instead starting with seed {start_seed}")

        if x0_bounds is None:
            x0_bounds = x_bounds

        x0_bounds = [(lb, ub,) if lb > x0_min else (x0_min, ub) for lb, ub in zip(x0_bounds.lb, x0_bounds.ub)]

        Parallel(n_jobs=n_jobs, verbose=100, backend=backend)(
            delayed(self._optimize_random_guess)(
                seed=seed,
                x_description=x_description,
                x_bounds=x_bounds,
                x0_bounds=x0_bounds,
                path=path,
                optimize_kwargs=optimize_kwargs
            )
            for seed in range(start_seed, start_seed + n_runs)
        )

    def _optimize_random_guess(self,
                               seed: int,
                               x_description: list[str],
                               x_bounds: Bounds,
                               x0_bounds: list[tuple[float, float]],
                               path: pathlib.Path,
                               optimize_kwargs):
        """Creates a random guess from a seed, and optimizes it"""
        # log uniform from scipy is not supported in 1.3.3
        def loguniform(lo, hi):
            return lo ** ((((log(hi) / log(lo)) - 1) * random()) + 1)

        x0 = np.array([loguniform(lb, ub) if ub > 0 else 0 for lb, ub in x0_bounds])
        path = path.joinpath(f'guess_{seed}/')
        path.mkdir()

        try:
            self.optimize(
                x0=x0,
                x_description=x_description,
                x_bounds=x_bounds,
                path=path,
                show_pbar=False,
                **optimize_kwargs
            )
        except InvalidPredictionError as e:
            warnings.warn(f"Invalid prediction was found at seed: {seed}: \n{e}")
            pass  # results are stored incase an error occurred due to self.optimize.

    @staticmethod
    def load_optimized_model(path: str | pathlib.Path) -> OptimizedModel:
        """Loads in the data from the log files.

        Parameters
        ----------
        path
            The path to the folder where the optimization progress has been stored.

        Returns
        -------
        :class:`OptimizedModel`
            A structured OptimizerProgress instance which contains all information that was logged.
        """
        return OptimizedModel(path)
