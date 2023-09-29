from abc import ABC, abstractmethod
from typing import Optional

from scipy.optimize import minimize
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
import os


class JSON_log:
    def __init__(self, path, mode="new"):
        self._path = path
        exists = os.path.isfile(path)

        if mode == "new":
            # create a new file
            if exists:
                raise FileExistsError("File already exists. To replace it use mode='replace'")
            with open(self._path, "w") as _:
                pass

        elif mode == "append":
            # append to the file
            if not exists:
                raise ValueError("File does not exist. Use mode='new' to create it.")

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

        self._all_X: pd.DataFrame = pd.DataFrame(list(df.loc[:, "x"]), columns=self.x_description)
        self._all_errors: pd.Series = df["error"]
        self._all_ratios: pd.Series = df["ratio"]
        self._all_times: pd.Series = df["datetime"]

        simplex = np.full((self.n_dimensions+1, self.n_dimensions), np.nan)
        sorted_errors = self._all_errors.sort_values(ascending=True)
        for n, index in enumerate(sorted_errors[:self.n_dimensions+1].keys()):
            simplex[n, :] = self._all_X.iloc[index, :].to_numpy()  # underscored variant so that no copying is required

        best_iteration_index = sorted_errors.index[0]

        self._best_X: pd.Series = pd.Series(self.all_X.loc[best_iteration_index, :], index=self.x_description)
        self.best_error: float = self.all_errors[best_iteration_index]
        self.best_ratio: float = self.all_errors[best_iteration_index]

    @property
    def best_X(self):
        return self._best_X.copy()

    @property
    def all_X(self):
        return self._all_X.copy()

    @property
    def all_errors(self):
        return self._all_errors.copy()

    @property
    def all_time_stamps(self):
        return self._all_times.copy()

    @property
    def all_ratios(self):
        return self._all_ratios.copy()


class RateConstantOptimizerTemplate(ABC):
    def __init__(self, raw_weights: list[tuple[str, float]]):
        self.raw_weights = raw_weights
        self.weights = None

    @abstractmethod
    def create_prediction(self, x: np.ndarray, x_description: list[str]) -> tuple[pd.DataFrame, float]:
        """
        Create a prediction of the system, given a set of parameters.
        :param x: Contains all parameters, which are to be optimized.
        Definitely includes are rate constants.
        :param x_description: Contains a description of each parameter.
        :returns: Predicted values of the concentration for each chemical, as a function of time.
                  Predicted ratio of compounds (enantiomeric ratio).
        """
        pass

    @abstractmethod
    def calculate_error_functions(self, pred: pd.DataFrame) -> pd.Series:
        """
        Calculate the error caused by each error function.
        :param pred: The predicted concentrations.
        :return errors: The unweighted errors of each error function.
        """
        pass

    def weigh_errors(self, errors: pd.Series, ) -> pd.Series:
        """
        weighs the errors
        :param errors: unweighted errors
        :return: weighed errors
        """
        if self.weights is None:
            weights = np.ones(errors.shape)
            for description, weight in self.raw_weights:
                index = errors.index.str.contains(description)
                if sum(index) == 0:
                    raise ValueError(f"no matches were found for {description}")
                weights[index] = weights[index] * weight
            self.weights = weights

        return errors * self.weights

    def calculate_total_error(self, errors):
        """
        weighs and sums the errors.
        :param errors: unweighted errors
        :return: weighed total error
        """
        return self.weigh_errors(errors).sum(skipna=False)

    def optimize(self,
                 x0: np.ndarray,
                 x_description: list[str],
                 bounds: list[tuple[float, float]],
                 path: str,
                 metadata: Optional[dict],
                 maxiter: float = 50000,
                 resume_from_simplex=None
                 ) -> None:
        """
        Optimizes the system, utilizing a nelder-mead algorithm.
        :param x0: Parameters which are to be optimized. Always contain the rate constants.
        :param x_description: Description of each parameter.
        :param bounds: A list containing tuples, which in turn contain the lower and upper bound for each parameter.
        :param path: Where the solution should be stored.
        :param metadata: The metadata that should be saved alongside the solution.
        :param maxiter: The maximum number of iterations.
        :param resume_from_simplex: When a simplex is given, the solution starts here.
        It can be used to resume the optimization process.
        """
        # enable logging of all information retrieved from the system
        log_path = f"{path}/optimization_log.json"
        if resume_from_simplex is None:  # new optimization progres
            logger = JSON_log(log_path)
            metadata_extended = {
                "raw_weights": self.raw_weights,
                "x0": x0,
                "x_description": x_description,
                "bounds": bounds,
                "maxiter": maxiter
            }
            if metadata is None:
                pass
            else:
                # overwrites the default meta data values
                for key, value in metadata.items():
                    metadata_extended[key] = value
            meta_data_log = JSON_log(f"{path}/settings_info.json", mode="new")
            meta_data_log.log(pd.Series(metadata_extended))
        else:
            logger = JSON_log(log_path, mode="append")

        def optimization_step(x):
            """The function is given a set of parameters by the Nelder-Mead algorithm.
            Proceeds to calculate the corresponding prediction and its total error.
            The results are stored in a log before the error is returned to the optimizer."""

            prediction, predicted_compound_ratio = self.create_prediction(x, x_description)
            errors = self.calculate_error_functions(prediction)
            total_error = self.calculate_total_error(errors)

            logger.log(pd.Series([x, total_error, predicted_compound_ratio], index=["x", "error", "ratio"]))
            return total_error

        def update_tqdm(_):
            """update the progress bar"""
            pbar.update(1)

        with tqdm(total=maxiter, miniters=25) as pbar:
            # the minimization process is stored within the log, containing all x's and errors.
            minimize(fun=optimization_step,
                     x0=x0,
                     method="Nelder-Mead",
                     bounds=bounds,
                     callback=update_tqdm,
                     options={"maxiter": maxiter, "disp": True, "adaptive": True, "return_all": False,
                              "initial_simplex": resume_from_simplex})

    @staticmethod
    def load_optimization_progress(path: str) -> OptimizerProgress:
        """
        Loads in the data from the log files.
        :param path: Folder in which the optimization_log.json and settings_info.json files can be found.
        :return optimizer progress: OptimizerProgress instance which contains all information that was logged.
        """
        return OptimizerProgress(path)
