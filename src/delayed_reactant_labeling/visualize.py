from __future__ import annotations

import os
import pathlib
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from delayed_reactant_labeling.predict import InvalidPredictionError
from delayed_reactant_labeling.optimize import RateConstantOptimizerTemplate, OptimizedModel
from collections.abc import Iterable


class VisualizeSingleModel:
    """
    Contains several methods which create plots of the :class:`OptimizedModel<optimize.OptimizedModel>`.
    Each method will save the figure where the name is equal to the function name.

    Parameters
    ----------
    path
        The path to the folder where the created images should be stored.
    model
        The model of which plots should be created.
    rate_constant_optimizer
        The user implemented class of :class:`optimize.RateConstantOptimizerTemplate`.
    plot_title
        The title (plt.Figure.suptitle) that will be given to each plot.
        If None (default), no title will be given.
    hide_params
        A boolean array, which indicate if the respective parameter should be hidden
        If None (default), all parameters will be shown.
    dpi
        The 'density per inch' that will be used when saving the images.
    extensions
        The file format(s) to which the image will be saved.
    overwrite_image
        If false (default), an FileExistsError will be raised if the image already exists. 
        If true, the image will be overwritten.
        
    Raises
    ------
    FileExistError
        Raised if the image already exists.
    """
    def __init__(self,
                 path: str | pathlib.Path,
                 model: OptimizedModel,
                 rate_constant_optimizer: RateConstantOptimizerTemplate,
                 plot_title: Optional[str] = None,
                 hide_params: Optional[np.ndarray] = None,
                 dpi: int = 600,
                 extensions: Optional[Iterable[str] | str] = None,
                 overwrite_image: bool = False):

        self.path = path if isinstance(path, pathlib.Path) else pathlib.Path(path)
        self.path.mkdir(exist_ok=True, parents=True)  # upon image creation we check for duplicates
        self.model = model
        self.RCO = rate_constant_optimizer
        self.plot_title = plot_title
        self.dpi = dpi
        extensions = [extensions] if isinstance(extensions, str) else extensions  # convert str to list
        self.extensions = ['png', 'svg'] if extensions is None else extensions
        self.overwrite_image = overwrite_image
        self.best_prediction = self.RCO.create_prediction(x=model.optimal_x.values, x_description=model.x_description)
        self.hide_params = np.zeros(len(model.x_description), dtype=bool) if hide_params is None else hide_params

    def _image_exists(self, name):
        """Raises an FileExistsError if an image already exist"""
        if self.overwrite_image:
            return

        for extension in self.extensions:
            if path := (self.path / f"{name}.{extension.split('.')[-1]}").exists():
                raise FileExistsError(f'An image already exists! \nPath: {path}\n')

    def _image_save(self, fig, name):
        fig.suptitle(self.plot_title)
        for extension in self.extensions:
            fig.savefig(self.path / f"{name}.{extension.split('.')[-1]}", dpi=self.dpi)  # remove leading .

    def plot_optimization_progress(self,
                              ratio: Optional[tuple[str, list[str]]] = None,
                              n_points: int = 100
                              ) -> tuple[plt.Figure, plt.Axes]:
        """Shows the error as function of the iteration number.

        Args
        ----
        ratio
            If None (default), only the error ratio will be shown.
            If given, the first element indicates the chemical of interest, and the second element the chemicals it is
            compared to. For example ('A', ['A', 'B']), calculates :math:`A / (A+B)`.
            It will plot the ratio for the last point in each prediction.
        n_points
            The number of iterations for which the ratio will be re-calculated.
            These are uniformly distributed over all possible iterations.
        """
        self._image_exists('optimization_progress')
        fig, ax = plt.subplots()
        ax.scatter(range(len(self.model.all_errors)), self.model.all_errors, alpha=0.3)

        if ratio is not None:
            ax2 = ax.twinx()
            found_ratio = []
            sample_points = np.linspace(0, len(self.model.all_x) - 1, n_points).round(0).astype(int)
            for sample in sample_points:
                pred = self.RCO.create_prediction(
                    x=self.model.all_x.iloc[sample, :],
                    x_description=self.model.all_x.columns)
                found_ratio.append((pred[ratio[0]].iloc[-1] / pred[ratio[1]].iloc[-1, :].sum()).mean())

            ax2.scatter(sample_points, found_ratio, alpha=0.3, color="C1")
            ax2.set_ylabel("ratio", color="C1")

        ax.set_xlabel("iteration")
        ax.set_ylabel("error", color="C0")
        self._image_save(fig, 'optimization_progress')
        return fig, ax

    def plot_x(self,
               *args: pd.Series,
               file_name: Optional[str] = None,
               group_as: Optional[list[str]] = None,
               show_remaining: bool = True,
               ) -> tuple[plt.Figure, plt.Axes]:
        """Plots the parameters, x.
        Args
        ----
        *args
            Parameters that should be compared against the models parameters.
            If the pd.Series is given a name, this will be used in the legend.
        file_name
            The file name for the image. This should not include any extension.
            If None (default), it is equal to 'plot_x'.
        group_as
            Group the parameters by a key.
            Each parameter can only be matched with one key exactly.
        show_remaining
            Show the parameter which were not matched by any key in group_as.
        """
        if file_name is None:
            file_name = 'plot_x'
        self._image_exists(file_name)

        if group_as is None:
            group_as = ['']

        data = [self.model.optimal_x]
        index = ['model']
        x_description = self.model.optimal_x.index
        for n, arg in enumerate(args):  # contain the same input
            assert set(arg.index) == set(x_description)
            data.append(arg)
            index.append(arg.name if not None else n+1)
        data = pd.DataFrame(data, index=index, columns=x_description)  # aligns the data

        key_hits = []
        for key in group_as:
            key_hits.append(x_description.str.contains(key))
        key_hits = pd.DataFrame(key_hits, columns=x_description, index=group_as)
        total_hits = key_hits.sum(axis=0)

        if any(total_hits > 1):
            raise ValueError(f'An item was matched by multiple keys.\n{total_hits.index[total_hits>1]}')

        if show_remaining and any(total_hits == 0):
            key_hits = pd.concat([key_hits, (total_hits == 0).to_frame('other').T])

        fig, axs = plt.subplots(len(key_hits), 1, squeeze=False)
        for ax, (group_key, selected_x) in zip(axs.flatten(), key_hits.iterrows()):
            ax.set_ylabel(group_key)
            data.loc[:, selected_x].T.plot.bar(ax=ax)

        for ax in axs.flatten()[1:]:
            ax.get_legend().remove()
        fig.supylabel('intensity')
        fig.tight_layout()
        self._image_save(fig, file_name)
        return fig, axs


    def show_optimization_path_in_pca(self, create_3d_video: bool = False, fps: int = 30) -> tuple[
            plt.Figure, list[plt.Axes]]:
        from sklearn.decomposition import PCA

        # explore pca space
        fig = plt.figure(figsize=(8, 8))
        gs = fig.add_gridspec(2, 2, width_ratios=(1, 4), height_ratios=(4, 1),
                              left=0.15, right=0.83, bottom=0.15, top=0.83,
                              wspace=0.05, hspace=0.05)
        ax = fig.add_subplot(gs[0, 1])

        pca = PCA().fit(X=self.progress.all_x)
        scattered = ax.scatter(self.progress.all_x.dot(pca.components_[0]),
                               self.progress.all_x.dot(pca.components_[1]),
                               c=np.arange(len(self.progress.all_x)))
        ax.tick_params(top=True, right=True, labeltop=True, labelright=True)
        ax_bbox = ax.get_position(original=True)
        cax = plt.axes([0.85, ax_bbox.ymin, 0.05, ax_bbox.size[1]])
        cbar = plt.colorbar(scattered, cax=cax)
        cbar.set_label("iteration")

        ax_pc0 = fig.add_subplot(gs[1, 1])
        ax_pc1 = fig.add_subplot(gs[0, 0])

        x = np.arange(sum(~self.hide_params))
        xticks = self.progress.x_description[~self.hide_params]

        ax_pc0.bar(x, pca.components_[0][~self.hide_params])
        ax_pc1.barh(x, pca.components_[1][~self.hide_params])

        ax_pc0.set_xlabel(f"component 0, explained variance {pca.explained_variance_ratio_[0]:.2f}")
        ax_pc0.set_xticks(x, xticks, rotation=90, fontsize="small")

        ax_pc1.set_ylabel(f"component 1, explained variance {pca.explained_variance_ratio_[1]:.2f}")
        ax_pc1.set_yticks(x, xticks, fontsize="small")

        def create_3d_video_animation():
            fig3d = plt.figure()
            ax3d = fig3d.add_subplot(projection="3d")

            ax3d.scatter(self.progress.all_x.dot(pca.components_[0]),
                         self.progress.all_x.dot(pca.components_[1]),
                         self.progress.all_x.dot(pca.components_[2]),
                         c=np.arange(len(self.progress.all_x)))
            ax3d.tick_params(bottom=False, left=False,
                             labelbottom=False, labelleft=False)

            # Rotate the axes and update
            files = []
            files_folder = f"{self.path}/pca_rotation_animation"
            os.makedirs(files_folder)

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
        fig.savefig(f"{self.path}path_in_pca_space.png", dpi=self.dpi)
        fig.savefig(f"{self.path}path_in_pca_space.svg", dpi=self.dpi)
        return fig, [ax, ax_pc0, ax_pc1]

    def show_error_contributions(self) -> tuple[plt.Figure, plt.Axes]:
        errors = self.rate_constant_optimizer.calculate_errors(self.best_prediction)
        weighed_errors = self.rate_constant_optimizer.weigh_errors(errors)
        fig, ax = plt.subplots()
        weighed_errors.plot.bar(ax=ax)
        ax.set_ylabel("MAE")
        ax.set_xlabel("rate constant")
        fig.tight_layout()
        fig.savefig(f"{self.path}bar_plot_of_all_errors.png", dpi=self.dpi)
        fig.savefig(f"{self.path}bar_plot_of_all_errors.svg", dpi=self.dpi)
        return fig, ax

    def show_enantiomer_ratio(self, intermediates: list[str], experimental: pd.DataFrame) -> tuple[plt.Figure, plt.Axes]:
        fig, ax = plt.subplots()
        ax.set_title(self.description)
        plotted_label = False
        for x, intermediate in enumerate(intermediates):
            total_pred = self.best_prediction[[f"{intermediate}{isomer}" for isomer in self.isomers]].sum(axis=1)
            for i, isomer in enumerate(self.isomers):
                if not plotted_label:
                    label = f"predicted {isomer}"
                else:
                    label = None
                y_pred = self.best_prediction[f"{intermediate}{isomer}"] / total_pred
                y_pred = y_pred[-60:].mean()
                ax.scatter(x, y_pred, label=label, marker="^", alpha=0.7, color=f"C{i}")
                ax.text(x + 0.07, y_pred, f"{y_pred * 100:.1f}%", ha="left", va="center")
            plotted_label = True

        plotted_label = False
        for x, intermediate in enumerate(intermediates):
            try:
                total_exp = experimental[[f"{intermediate}{isomer}" for isomer in self.isomers]].sum(
                    axis=1)
                for i, isomer in enumerate(self.isomers):
                    if not plotted_label:
                        label = f"experimental {isomer}"
                    else:
                        label = None
                    y_exp = experimental[f"{intermediate}{isomer}"] / total_exp
                    y_exp = y_exp[-60:].mean()
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
        fig.savefig(f"{self.path}enantiomer_ratio.png", dpi=self.dpi)
        fig.savefig(f"{self.path}enantiomer_ratio.svg", dpi=self.dpi)
        return fig, ax

    def animate_rate_over_time(
            self,
            n_frames=300,
            fps=30
    ) -> None:
        from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

        n_iter = self.progress.n_iterations
        if n_frames > n_iter:
            raise ValueError("specified to analyze more frames than available iterations")

        fig, (ax_rates, ax_error) = plt.subplots(2)

        ticks = self.progress.all_x.columns[~self.hide_params]
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
            rates = self.progress.all_x.loc[i, :]
            ax_rates.clear()
            ax_rates.bar(x, rates.iloc[self.hide_params])

            ax_rates.set_xticks(x, ticks, rotation=90, fontsize="small")
            ax_rates.set_ylabel("k")
            ax_rates.legend(loc=1)
            ax_rates.set_title("found rate")

            # update the error plot
            ax_error.scatter(i, self.progress.all_errors[i], color="tab:blue")

            ax_error.set_xlabel("iteration")
            ax_error.set_ylabel("MAE", color="tab:blue")
            fig.suptitle(f"{self.description}")

            file = f"{files_folder}/frame_{i}.jpg"
            files.append(file)
            fig.tight_layout()
            fig.savefig(file)

        clip = ImageSequenceClip(files, fps=fps)
        clip.write_videofile(f"{self.path}visualized_rate_over_time.mp4")

    def show_rate_sensitivity(self,
                              lower_power: int = -6,
                              upper_power: int = 2,
                              steps: int = 101,
                              threshold=3) -> (plt.Figure, plt.Axes):
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        x_value = np.logspace(lower_power, upper_power, steps)

        ticks = self.progress.optimal_x[~self.hide_params]
        errors = np.full((len(x_value), len(ticks)), np.nan)

        # loop over all non-constant values and adjust those
        for col, key in enumerate(tqdm(self.progress.optimal_x[~self.hide_params].keys())):
            for row, adjusted_x in enumerate(x_value):
                # insert all values into the plot
                best_X = self.progress.optimal_x.copy()
                best_X[key] = adjusted_x
                try:
                    prediction = self.rate_constant_optimizer.create_prediction(
                        x=best_X.to_numpy(), x_description=self.progress.x_description)
                    unweighed_error = self.rate_constant_optimizer.calculate_errors(prediction)
                    found_error = self.rate_constant_optimizer.calculate_total_error(unweighed_error)
                except InvalidPredictionError:
                    found_error = np.nan
                errors[row, col] = found_error

        min_error = np.nanmin(errors)
        errors[errors > threshold * min_error] = threshold * min_error

        fig, ax = plt.subplots()
        im = ax.imshow(errors, origin="lower", aspect="auto")

        ax.set_xticks(np.arange(len(ticks)), ticks.index, fontsize="small")
        ax.tick_params(axis='x', rotation=45)

        n_orders_of_magnitude = int(upper_power - lower_power + 1)
        ind = np.linspace(0, len(x_value) - 1, n_orders_of_magnitude)  # .round(0).astype(int)
        yticks = [f"{y:.0e}" for y in np.logspace(lower_power, upper_power, n_orders_of_magnitude)]
        ax.set_yticks(ind, yticks)
        ax.set_ylabel("adjusted value")

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", "5%", pad="3%")
        fig.colorbar(im, cax=cax, label="MAE")
        ax.set_title(f"{self.description}")
        fig.tight_layout()
        fig.savefig(f"{self.path}sensitivity_of_rate.png", dpi=self.dpi)
        fig.savefig(f"{self.path}sensitivity_of_rate.svg", dpi=self.dpi)

        return fig, ax


class VisualizeMultipleSolutions:
    def __init__(self, path: pathlib.Path, max_guess: int = np.inf):
        """Loads the data in the path."""
        self.complete_all_X = []
        self.complete_initial_X = []
        self.complete_optimal_X = []
        self.complete_found_error = []
        self.x_description = None

        for n, guess_path in tqdm(enumerate(path.iterdir())):
            if not guess_path.is_dir():
                max_guess += 1
                continue
            if n > max_guess:
                break

            progress = OptimizedModel(guess_path)

            self.complete_all_X.append(progress.all_x)
            self.complete_initial_X.append(progress.all_x.iloc[0, :])
            self.complete_optimal_X.append(progress.optimal_x)
            self.complete_found_error.append(progress.optimal_error)

            if self.x_description is None:
                self.x_description = progress.x_description

        # complete all X is not here as it is already a 2D matrix
        self.complete_initial_X = np.array(self.complete_initial_X)
        self.complete_optimal_X = np.array(self.complete_optimal_X)
        self.complete_found_error = np.array(self.complete_found_error)

    def show_error_all_runs(self, top_n=-1) -> tuple[plt.Figure, plt.Axes]:
        if top_n == -1:
            top_n = len(self.complete_found_error)
        ind = np.argsort(self.complete_found_error)[:top_n]

        fig, ax = plt.subplots(layout='tight')
        ax.scatter(np.arange(top_n), self.complete_found_error[ind])
        ax.set_xlabel('run number (sorted by error)')
        ax.set_ylabel('error')
        return fig, ax

    def show_summary_all_runs(self,
                              rco: RateConstantOptimizerTemplate,
                              compound_ratio: list[str, list[str]],
                              top_n: int = -1,
                              max_error: float = 0.5) -> tuple[plt.Figure, plt.Axes]:
        fig, ax = plt.subplots(layout='tight')
        ind = np.argsort(self.complete_found_error)[:top_n]

        found_ratio = []
        for run_index in ind:
            pred = rco.create_prediction(self.complete_optimal_X[run_index].values, self.complete_optimal_X[run_index].index)
            found_ratio.append(pred[compound_ratio[0]] / pred[compound_ratio[1]].sum(axis=1))  # TODO double check this

        error = np.array(self.complete_found_error)
        error[error > max_error] = max_error
        im = ax.scatter(np.arange(len(ind)),
                        found_ratio[ind],
                        c=np.array(self.complete_found_error)[ind])

        fig.colorbar(im, ax=ax, label='error')
        ax.set_xlabel('run number (sorted by error)')
        ax.set_ylabel('ratio')
        return fig, ax

    def show_rate_constants(self, max_error: float, index_constant_values: np.ndarray = None) -> tuple[plt.Figure, plt.Axes]:
        ind_allowed_error = np.array(self.complete_found_error) < max_error
        df = pd.DataFrame(np.array(self.complete_optimal_X)[ind_allowed_error], columns=self.x_description)
        if index_constant_values is not None:
            df = df.loc[:, ~index_constant_values]

        fig, ax = plt.subplots(layout="tight")
        ax.boxplot(df)
        ax.set_title("distribution of optimal rates")
        _ = ax.set_xticks(1 + np.arange(len(df.columns)), df.columns, rotation=90)
        ax.set_yscale("log")
        ax.set_ylabel("value of k")
        return fig, ax

    def show_biplot(self,
                    max_error: float,
                    pc1: int = 0,
                    pc2: int = 1,
                    ax: plt.Axes = None):
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        data = []
        for error, initial, optimal in zip(self.complete_found_error, self.complete_initial_X, self.complete_optimal_X):
            if error > max_error:
                continue
            data.append(initial)
            data.append(optimal)

        data = pd.DataFrame(data).reset_index(drop=True)  # by converting to df first, we assure that the columns are aligned
        scaler = StandardScaler()
        scaler.fit(X=data.to_numpy())  # scale each rate so its std deviation becomes 1
        pca = PCA().fit(X=scaler.transform(data.to_numpy()))

        # scores
        for error, initial, optimal in zip(self.complete_found_error, self.complete_initial_X, self.complete_optimal_X):
            if error > max_error:
                continue
            ax.scatter(
                x=scaler.transform(initial.to_numpy().reshape(1, -1)).dot(pca.components_[pc1]),
                y=scaler.transform(initial.to_numpy().reshape(1, -1)).dot(pca.components_[pc2]),
                marker='o', color='tab:blue'
            )
            ax.scatter(
                x=scaler.transform(optimal.to_numpy().reshape(1, -1)).dot(pca.components_[pc1]),
                y=scaler.transform(optimal.to_numpy().reshape(1, -1)).dot(pca.components_[pc2]),
                marker='*', color='tab:orange'
            )
        ax.scatter(np.nan, np.nan, color="tab:blue", marker="o", label="initial")
        ax.scatter(np.nan, np.nan, color="tab:orange", marker="*", label="optimal")
        ax.legend()

        # maximize the size of the loadings
        x_factor = abs(np.array(ax.get_xlim())).min() / pca.components_[pc1].max()
        y_factor = abs(np.array(ax.get_ylim())).min() / pca.components_[pc2].max()

        # loadings
        for rate, loading1, loading2 in zip(data.columns, pca.components_[pc1], pca.components_[pc2]):
            ax.plot([0, loading1*x_factor], [0, loading2*y_factor], color='tab:gray')
            ax.text(loading1*x_factor, loading2*y_factor, rate, ha='center', va='bottom')

        ax.set_title('biplot')
        ax.set_xlabel(f'PC {pc1}, explained variance {pca.explained_variance_ratio_[pc1]:.2f}')
        ax.set_ylabel(f'PC {pc2}, explained variance {pca.explained_variance_ratio_[pc2]:.2f}')

        return fig, ax
