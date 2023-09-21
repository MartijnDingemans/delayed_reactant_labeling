import os
from typing import Optional, Callable

import matplotlib.pyplot as plt
import scipy.optimize as opt
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.decomposition import PCA
from tqdm import tqdm
from delayed_reactant_labeling.delayed_reactant_labeling import Experimental_Conditions


class JSON_log:
    def __init__(self, path, mode="new"):
        self._path = path
        exists = os.path.isfile(path)

        if mode == "new":
            # create a new file
            if exists:
                raise ValueError("File already exists. To replace it use mode='replace'")
            with open(self._path, "w") as f:
                pass

        elif mode == "append":
            # append to the file
            if not exists:
                raise ValueError("File does not exist. Use mode='new' to create it.")

        elif mode == "replace":
            # replace the file
            with open(self._path, "w") as f:
                pass

    def log(self, data: pd.Series):
        data["datetime"] = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        with open(self._path, "a") as f:
            f.write(data.to_json() + "\n")


def optimize_by_nelder_mead(
        rate_constants_names: list[str],
        logger: JSON_log,
        calculate_error_func: Callable[[pd.Series], tuple[float, float]],
        vertex: np.array,
        bounds: list[tuple[float, float]],
        maxiter: float = 200000,
        simplex: np.array = None,
        manipulate_rate: Callable[[pd.Series], pd.Series] = None) -> None:
    """
    :param rate_constants_names: rate constants which describe the reaction. Used not for the values but for the construction
    of the initial vertex.
    :param logger: Logger which saves the results during the optimization process. Allows for interruption during calc.
    :param calculate_error_func: Function which calculates the error for a system when provided with the rate constant.
    :param vertex: The simplex will overwrite the initial vertex if given.
    :param bounds: The boundaries of the system
    :param maxiter: max iterations
    :param simplex: initial simplex, when given the optimizer resumes from this position.
    :param manipulate_rate: Function which modifies the rates before the error calculation. It can be used to fix values.
    :return:
    """

    # Wrapper function. Stores all values in the log and allows for manipulating the rate constants before calculating.
    def f(x):
        x = pd.Series(x, index=rate_constants_names)
        if manipulate_rate is not None:
            x = manipulate_rate(x)

        error, ratio = calculate_error_func(x)
        logger.log(pd.Series([x, error, ratio], index=["x", "error", "ratio"]))
        return error

    def update_tqdm(xk):
        pbar.update(1)

    with tqdm(total=maxiter) as pbar:
        # the minimization process is stored within the log, containing all x's and errors.
        opt.minimize(fun=f, x0=vertex, method="Nelder-Mead", bounds=bounds,
                     options={"maxiter": maxiter, "disp": True, "adaptive": True, "return_all": False,
                              "initial_simplex": simplex}, callback=update_tqdm)


def feature_selecting_optimization_by_nelder_mead(
        path: str,
        description: str,
        calculate_error_func: Callable[[pd.Series], tuple[float, float]],
        initial_vertex: np.array,
        initial_bounds: list[tuple[float, float]],
        experimental: pd.DataFrame,
        experimental_conditions: Experimental_Conditions,
        calculate_individual_errors: Callable,
        create_prediction: Callable,
        rate_constants_names: list[str],
        literature_rate_constants: [pd.Series],
        isomers: list[str],
        max_iter: int = 6000,
        min_dimensions: int = 8,
        manipulate_rate: Callable[[pd.Series], pd.Series] = None
):
    """
    All results will be stored in a folder with the current datetime to avoid accidents.
    1. The model creates a prediction using all available rate constants.
    2. The rate constants will be optimized.
    3. For each rate constant a permutation will be made, where it has half of its usual value.
    4. For the rate constant where this impacted the error the least [minimal error]
    5. Remove it from the system by settings its bounds to 0
    6. Go back to 1., until the desired amount of dimensions has been reached.
    """

    # extend the given path with a name
    top_path = f"{path}_iterative_selection"
    os.mkdir(top_path)
    feature_log = JSON_log(f"{top_path}/feature_optimization_log.json")

    relevant_k = [ub > 0 for lb, ub in initial_bounds]
    current_dimensions = sum(relevant_k)

    relevant_k = pd.Series(relevant_k, index=rate_constants_names)
    bounds = initial_bounds
    vertex = initial_vertex
    iteration = 0
    while current_dimensions > min_dimensions:
        iteration_data = {"iteration": iteration,
                          "bounds": bounds,
                          "vertex": vertex}

        # prepare for optimization
        sub_path = f"{top_path}/iteration_{iteration}"
        os.mkdir(sub_path)
        log_path = f"{sub_path}/iteration_{iteration}_optimization_log.json"

        # Optimize the available model
        optimize_by_nelder_mead(
            rate_constants_names=rate_constants_names,
            logger=JSON_log(log_path),
            calculate_error_func=calculate_error_func,
            bounds=bounds,
            vertex=vertex,
            maxiter=max_iter,
            manipulate_rate=manipulate_rate)

        # load the model and permute all k values
        progress = load_nelder_mead_progress(rate_constants_names=rate_constants_names, log_path=log_path)
        error_without_k = {}
        for k, rate in progress.best_rates.items():
            # for k's which have an upper bound of 0, (no longer relevant!) skip the entry
            if not relevant_k[k]:
                error_without_k[k] = np.inf
                continue

            adjusted_rates = progress.best_rates.copy()
            adjusted_rates[k] = rate * 0

            error_without_k[k] = calculate_error_func(adjusted_rates)[0]
        error_without_k = pd.Series(error_without_k)
        iteration_data["MAE_optimized"] = progress.best_error
        iteration_data["rates_optimized"] = progress.best_rates
        iteration_data["error_without_k"] = error_without_k

        # the k value which impacted the error the least upon halving is deemed the least useful
        ind = error_without_k.argmin()
        iteration_data["worst_k"] = {"k": error_without_k.index[ind], "MAE": error_without_k.iloc[ind]}

        # set its bounds to zero, so it will be neglected in future iterations
        # proceed the optimization process from the current local min.
        vertex = progress.best_rates
        vertex[ind] = 0
        bounds[ind] = (0, 0,)
        relevant_k.iloc[ind] = False

        # log all information
        iteration_data = pd.Series(iteration_data)
        feature_log.log(data=iteration_data)
        current_dimensions -= 1

        # visualize all information
        vrc = Visualize_Rate_Constants(
            path=f"{sub_path}/",
            description=description,
            rate_names=rate_constants_names,
            progress=progress,
            create_prediction=create_prediction,
            calculate_individual_error=calculate_individual_errors,
            experimental=experimental,
            experimental_conditions=experimental_conditions,
            isomers=isomers,
        )
        vrc.show_error_over_time()
        vrc.show_optimization_path_in_pca(create_3d_video=False)
        vrc.show_comparison_with_literature(desired_k=["k1", "k2", "k-2", "k3", "k4"],
                                            literature_rate_constants=literature_rate_constants)
        vrc.show_error_contributions()
        vrc.show_isomer_and_label_ratio()
        vrc.animate_rate_over_time()
        vrc.show_enantiomer_ratio(intermediates=["3", "4", "5", "6"])

        iteration += 1
    print("optimization ended")


class load_nelder_mead_progress:
    def __init__(self, rate_constants_names: list[str], log_path: str, last_n: int = 200):
        try:
            df = pd.read_json(log_path, lines=True)
        except ValueError as e:
            print("an error exists in the code. Perhaps manually a line was removed without re-adding a new line.")
            raise e

        simplex = []
        df_last_n = df.iloc[-last_n:, :]
        for row in df_last_n.loc[df_last_n["error"].sort_values()[:len(rate_constants_names) + 1].index, "x"]:
            simplex.append(np.array(row))
        self.simplex = np.array(simplex)

        best_iteration = df_last_n.iloc[df_last_n['error'].argmin(), :]
        self._best_rates: pd.Series = pd.Series(best_iteration["x"], index=rate_constants_names)
        self.best_error: float = best_iteration["error"]
        self.best_ratio: float = best_iteration["ratio"]

        self.n_iter = len(df)
        all_iterations = []
        for row in df.loc[:, "x"]:
            all_iterations.append(pd.Series(row))

        self._all_rates: pd.DataFrame = pd.DataFrame(all_iterations, columns=rate_constants_names)
        self._all_errors: pd.Series = df["error"]
        self._all_ratios: pd.Series = df["ratio"]
        self._all_times: pd.Series = df["datetime"]

    def __len__(self):
        return self.n_iter

    @property
    def best_rates(self):
        return self._best_rates.copy()

    @property
    def all_rates(self):
        return self._all_rates.copy()

    @property
    def all_errors(self):
        return self._all_errors.copy()

    @property
    def all_times(self):
        return self._all_times.copy()

    @property
    def all_ratios(self):
        return self._all_ratios.copy()


class Visualize_Rate_Constants:
    def __init__(self,
                 path: str,
                 description: str,
                 rate_names: list[str],
                 progress: load_nelder_mead_progress,
                 create_prediction: Optional[Callable] = None,
                 calculate_individual_error: Optional[Callable] = None,
                 calculate_total_error: Optional[Callable] = None,
                 experimental: Optional[pd.DataFrame] = None,
                 experimental_conditions: Experimental_Conditions = None,
                 isomers: Optional[list[str]] = None,
                 ):
        self.path = path
        self.description = description
        self.progress = progress

        self._prediction = None
        self.create_prediction = create_prediction
        self.calculate_individual_error = calculate_individual_error
        self.calculate_total_error = calculate_total_error
        self.experimental = experimental
        self.experimental_conditions = experimental_conditions

        self.isomers = isomers
        self.rate_names = rate_names
        self.index_reverse_reaction = np.array([True if "k-" in k else False for k in self.rate_names])

    @property
    def prediction(self):
        if self._prediction is None:
            # recompute the best prediction so that we can make plots of it.
            self.experimental, self._prediction = self.create_prediction(_rate_constants=self.progress.best_rates)
        return self._prediction

    def show_error_over_time(self) -> (plt.Figure, plt.Axes):
        # explore the solution
        fig, ax = plt.subplots()
        ax.scatter(range(1, 1 + len(self.progress.all_errors)), self.progress.all_errors, alpha=0.3)

        ax2 = ax.twinx()
        ax2.scatter(range(1, 1 + len(self.progress.all_ratios)), self.progress.all_ratios, alpha=0.3, color="C1")
        ax2.set_ylabel("ratio", color="C1")

        ax.set_xlabel("iteration")
        ax.set_ylabel("sum of MAE", color="C0")
        ax.set_title(f"{self.description}")
        fig.show()
        fig.savefig(f"{self.path}error_over_time.png", dpi=1000)
        fig.savefig(f"{self.path}error_over_time.svg", dpi=1000)
        return fig, ax

    def show_comparison_with_literature(self,
                                        desired_k: list[str],
                                        literature_rate_constants: pd.Series) -> (plt.Figure, plt.Axes):
        """
        Show a comparison between the found rates and some given rates. Rate name should be {k}_{isomer}.
        :param desired_k: on which rate constants
        :param literature_rate_constants: Series containing the rate constants, has to follow exact same naming.
        :return: figure, ax
        """
        fig, axs = plt.subplots(3, figsize=(8, 8))
        for ax, isomer in zip(axs, self.isomers):
            rate_found = []
            rate_lit = []

            for k in desired_k:
                rate_found.append(self.progress.best_rates[f"{k}_{isomer}"])
                rate_lit.append(literature_rate_constants[f"{k}_{isomer}"])

            x = np.arange(len(rate_found))
            multiplier = -0.5
            width = 0.4
            max_val = max([max(rate_found), max(rate_lit)])
            settings = {"ha": "center", "fontweight": "bold"}
            for vals, descr in zip([rate_found, rate_lit], ["found", "lit."]):
                ax.bar(x + width * multiplier,
                       vals,
                       width,
                       label=descr)
                for val, val_x in zip(vals, x + width * multiplier):
                    if val > 0.005:
                        form = f"{val:.3f}"
                    else:
                        form = f"{val:.0e}"

                    if val < 0.5 * max_val:
                        ax.text(val_x, val + 0.02 * max_val, form, color="k", **settings)
                    else:
                        ax.text(val_x, val - 0.09 * max_val, form, color="w", **settings)
                multiplier += 1

            ax.set_xticks(x, desired_k, rotation=90, fontsize="small")
        axs[-1].legend()
        fig.suptitle(f"{self.description}")
        fig.tight_layout()
        fig.show()
        fig.savefig(f"{self.path}comparison_with_literature.png", dpi=1000)
        fig.savefig(f"{self.path}comparison_with_literature.svg", dpi=1000)
        return fig, axs

    def show_optimization_path_in_pca(self, create_3d_video: bool = False, fps: int = 30) -> (
            plt.Figure, list[plt.Axes]):
        # explore pca space
        fig = plt.figure(figsize=(8, 8))
        gs = fig.add_gridspec(2, 2, width_ratios=(1, 4), height_ratios=(4, 1),
                              left=0.15, right=0.83, bottom=0.15, top=0.83,
                              wspace=0.05, hspace=0.05)
        ax = fig.add_subplot(gs[0, 1])

        pca = PCA().fit(X=self.progress.all_rates)
        scattered = ax.scatter(self.progress.all_rates.dot(pca.components_[0]),
                               self.progress.all_rates.dot(pca.components_[1]),
                               c=np.arange(len(self.progress.all_rates)))
        ax.tick_params(bottom=False, left=False,
                       labelbottom=False, labelleft=False)
        ax_bbox = ax.get_position(original=True)
        cax = plt.axes([0.85, ax_bbox.ymin, 0.05, ax_bbox.size[1]])
        cbar = plt.colorbar(scattered, cax=cax)
        cbar.set_label("iteration")

        ax_pc0 = fig.add_subplot(gs[1, 1])
        ax_pc1 = fig.add_subplot(gs[0, 0])

        x = np.arange(len(pca.components_[0][self.index_reverse_reaction]))
        ticks = self.progress.best_rates[~self.index_reverse_reaction].index
        multiplier = -0.5
        width = 0.4
        for ind, descr in [(~self.index_reverse_reaction, "forward",),
                           (self.index_reverse_reaction, "reverse",)]:
            ax_pc0.bar(x + width * multiplier,
                       pca.components_[0][ind],
                       width,
                       label=descr)
            ax_pc1.barh(x + width * multiplier,
                        pca.components_[1][ind],
                        width,
                        label=descr)
            multiplier += 1

        ax_pc0.set_xlabel(f"component 0, explained variance {pca.explained_variance_ratio_[0]:.2f}")
        ax_pc0.set_xticks(x, ticks, rotation=90, fontsize="small")
        ax_pc0.tick_params(left=False, labelleft=False)

        ax_pc1.set_ylabel(f"component 1, explained variance {pca.explained_variance_ratio_[1]:.2f}")
        ax_pc1.set_yticks(x, ticks, fontsize="small")
        ax_pc1.tick_params(bottom=False, labelbottom=False)

        def create_3d_video_animation():
            fig3d = plt.figure()
            ax3d = fig3d.add_subplot(projection="3d")

            ax3d.scatter(self.progress.all_rates.dot(pca.components_[0]),
                         self.progress.all_rates.dot(pca.components_[1]),
                         self.progress.all_rates.dot(pca.components_[2]),
                         c=np.arange(len(self.progress.all_rates)))
            ax3d.tick_params(bottom=False, left=False,
                             labelbottom=False, labelleft=False)

            # sub-function is used to return None to skip the remainder
            # Rotate the axes and update
            files = []
            files_folder = f"{self.path}/pca_rotation_animation"
            try:
                os.makedirs(files_folder)
            except FileExistsError:
                print("Already a folder containing the pca_rotation exists. skipping...")
                return None

            for n, angle in tqdm(enumerate(range(0, 360 * 4 + 1))):
                # Normalize the angle to the range [-180, 180] for display
                angle_norm = (angle + 180) % 360 - 180

                # Cycle through a full rotation of elevation, then azimuth, roll, and all
                elev = azim = roll = 0
                if angle <= 360:
                    elev = angle_norm
                elif angle <= 360 * 2:
                    azim = angle_norm
                elif angle <= 360 * 3:
                    roll = angle_norm
                else:
                    elev = azim = roll = angle_norm

                # Update the axis view and title
                ax3d.view_init(elev, azim, roll)
                ax3d.set_title('Elevation: %d°, Azimuth: %d°, Roll: %d°' % (elev, azim, roll))
                file = f"{files_folder}/{n}.jpg"
                files.append(file)
                fig3d.savefig(file)

            import moviepy.video.io.ImageSequenceClip
            clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(files, fps=fps)
            clip.write_videofile(f'{self.path}pca_rotation.mp4')

        if create_3d_video:
            create_3d_video_animation()

        fig.suptitle(f"{self.description}")
        fig.show()
        fig.savefig(f"{self.path}path_in_pca_space.png", dpi=1000)
        fig.savefig(f"{self.path}path_in_pca_space.svg", dpi=1000)
        return fig, [ax, ax_pc0, ax_pc1]

    def show_error_contributions(self) -> (plt.Figure, plt.Axes):
        total_error, ratio, fig, ax = self.calculate_total_error(self.progress.best_rates, create_plot=True)
        ax.set_title(f"{self.description}\ntotal MAE: {total_error:.3f}")
        fig.tight_layout()
        fig.show()
        fig.savefig(f"{self.path}bar_plot_of_all_errors.png", dpi=1000)
        fig.savefig(f"{self.path}bar_plot_of_all_errors.svg", dpi=1000)
        return fig, ax

    def show_error_contributions_detailed(self) -> ((plt.Figure, plt.Axes), (plt.Figure, plt.Axes), (plt.Figure, plt.Axes)):
        prediction = self.prediction
        _experimental = self.experimental

        all_errors, (fig_iso, axs_iso), (fig_lab, axs_lab), (fig_TIC, axs_TIC) = self.calculate_individual_error(
            _experimental,
            prediction,
            True)

        for ax in axs_iso:
            ax.legend()
        fig_iso.suptitle(self.description)
        fig_iso.show()
        fig_iso.savefig(f"{self.path}fit_between_isomers.png", dpi=1000)
        fig_iso.savefig(f"{self.path}fit_between_isomers.svg", dpi=1000)

        for ax in axs_lab:
            ax.legend()
        fig_lab.suptitle(self.description)
        fig_lab.show()
        fig_lab.savefig(f"{self.path}fit_with_labeled.png", dpi=1000)
        fig_lab.savefig(f"{self.path}fit_with_labeled.svg", dpi=1000)

        fig_TIC.suptitle(self.description)
        fig_TIC.show()
        fig_TIC.savefig(f"{self.path}fit_with_TIC.png", dpi=1000)
        fig_TIC.savefig(f"{self.path}fit_with_TIC.svg", dpi=1000)

        return (fig_iso, axs_iso,), (fig_lab, axs_lab,), (fig_TIC, axs_TIC,)

    def show_enantiomer_ratio(self, intermediates: list[str]) -> (plt.Figure, plt.Axes):
        fig, ax = plt.subplots()
        ax.set_title(self.description)
        plotted_label = False
        for x, intermediate in enumerate(intermediates):
            total_pred = self.prediction[[f"{intermediate}{isomer}" for isomer in self.isomers]].sum(axis="columns")
            for i, isomer in enumerate(self.isomers):
                if not plotted_label:
                    label = f"predicted {isomer}"
                else:
                    label = None
                y_pred = self.prediction[f"{intermediate}{isomer}"].divide(total_pred).iloc[-60:].mean()
                ax.scatter(x, y_pred, label=label, marker="^", alpha=0.7, color=f"C{i}")
                ax.text(x + 0.07, y_pred, f"{y_pred * 100:.1f}%", ha="left", va="center")
            plotted_label = True

        plotted_label = False
        for x, intermediate in enumerate(intermediates):
            try:
                total_exp = self.experimental[[f"{intermediate}{isomer}" for isomer in self.isomers]].sum(
                    axis="columns")
                for i, isomer in enumerate(self.isomers):
                    if not plotted_label:
                        label = f"experimental {isomer}"
                    else:
                        label = None
                    y_exp = self.experimental[f"{intermediate}{isomer}"].divide(total_exp).iloc[-60:].mean()
                    ax.scatter(x, y_exp, label=label, marker="x", alpha=0.7, color=f"C{i}")
                plotted_label = True
            except KeyError:
                print(f"could not find {intermediate} in the experimental data. skipping...")

        ax.set_ylabel("isomer fraction")
        ax.legend()
        ax.set_xticks(range(len(intermediates)), intermediates)
        ax.set_xlabel("compound")
        xl, xu = ax.get_xlim()
        ax.set_xlim(xl, xu + 0.2)
        fig.tight_layout()
        fig.show()
        fig.savefig(f"{self.path}enantiomer_ratio.png", dpi=1000)
        fig.savefig(f"{self.path}enantiomer_ratio.svg", dpi=1000)
        return fig, ax

    def animate_rate_over_time(
            self,
            n_frames=300,
            fps=30
    ):
        import moviepy.video.io.ImageSequenceClip

        n_iter = self.progress.n_iter
        if n_frames > n_iter:
            raise ValueError("specified to analyze more frames than available iterations")

        fig, (ax_rates, ax_error) = plt.subplots(2)
        ax_ratio = ax_error.twinx()

        ticks = self.progress.all_rates.columns[~self.index_reverse_reaction]
        x = np.arange(len(ticks))

        ax_error.set_xlim(0, n_iter - 1)
        files = []
        files_folder = f"{self.path}/animation_rate_over_time"
        try:
            os.makedirs(files_folder)
        except FileExistsError:
            print("Already a folder containing the pca_rotation exists. skipping...")
            return None

        for i in tqdm(np.linspace(0, n_iter - 1, n_frames).round().astype(int)):
            # rate plot
            rates = self.progress.all_rates.loc[i, :]
            multiplier = -0.5
            width = 0.4
            ax_rates.clear()
            for ind, descr in [(~self.index_reverse_reaction, "forward",),
                               (self.index_reverse_reaction, "reverse",)]:
                ax_rates.bar(x + width * multiplier,
                             rates.iloc[ind],
                             width,
                             label=descr)
                multiplier += 1

            ax_rates.set_xticks(x, ticks, rotation=90, fontsize="small")
            ax_rates.set_ylabel("k")
            # ax_rates.set_ylim(0, 8)
            ax_rates.legend(loc=1)
            ax_rates.set_title("found rate")

            # update the error plot
            ax_error.scatter(i, self.progress.all_errors[i], color="tab:blue")
            ax_ratio.scatter(i, self.progress.all_ratios[i], color="tab:orange")

            ax_error.set_xlabel("iteration")
            ax_error.set_ylabel("MAE", color="tab:blue")
            ax_ratio.set_ylabel("ratio", color="tab:orange")
            fig.suptitle(f"{self.description}")

            file = f"{files_folder}/frame_{i}.jpg"
            files.append(file)
            fig.tight_layout()
            fig.savefig(file)

        clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(files, fps=fps)
        clip.write_videofile(f"{self.path}visualized_rate_over_time.mp4")

    def show_rate_sensitivity(self,
                              k_multiplier=np.linspace(0.5, 1.5, 11),
                              threshold=5) -> (plt.Figure, plt.Axes):
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        ticks = self.progress.best_rates.copy()  # when a rate is zero, remove it
        col = 0  # tracker here because some columns get skipped
        errors = np.full((len(k_multiplier), len(self.progress.best_rates)), np.nan)
        for (k, rate) in tqdm(self.progress.best_rates.items()):
            if rate == 0:
                del ticks[k]
                continue

            adjusted_rates = k_multiplier * rate
            for row, adjusted_rate in enumerate(adjusted_rates):
                rates = self.progress.best_rates.copy()
                rates[k] = adjusted_rate
                errors[row, col] = self.calculate_total_error(rates)[0]
            col += 1

        fig, ax = plt.subplots()
        errors[errors > threshold * errors.min()] = threshold * errors.min()
        im = ax.imshow(errors, origin="lower")

        ax.set_xticks(np.arange(len(ticks)), ticks.index, fontsize="small")
        ax.tick_params(axis='x', rotation=45)

        ind = np.linspace(0, len(k_multiplier)-1, 5).round(0).astype(int)
        ax.set_yticks(ind, k_multiplier[ind].round(2))

        ax.set_ylabel("multiplier of k")

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", "5%", pad="3%")
        fig.colorbar(im, cax=cax, label="MAE")
        ax.set_title(f"{self.description}")
        fig.tight_layout()
        fig.show()
        fig.savefig(f"{self.path}sensitivity_of_rate.png", dpi=1000)
        fig.savefig(f"{self.path}sensitivity_of_rate.svg", dpi=1000)

        return fig, ax
