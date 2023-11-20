from __future__ import annotations

import pathlib
import warnings
from collections.abc import Iterable
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from tqdm import tqdm


from delayed_reactant_labeling.optimize import RateConstantOptimizerTemplate, OptimizedModel
from delayed_reactant_labeling.predict import InvalidPredictionError


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

    def _image_exists(self, file_name):
        """Raises an FileExistsError if an image already exist"""
        if self.overwrite_image:
            return

        for extension in self.extensions:
            path = self.path / f"{file_name}.{extension.split('.')[-1]}"
            if path.exists():
                raise FileExistsError(f'An image already exists! \nPath: {path}\n')

    def save_image(self, fig, file_name, tight_layout=True):
        """Saves an image with all relevant extensions, and the correct dpi.
        It will be stored in the folder, specified by the ``path``.

        Args
        ----
        fig
            The figure that is to be saved.
        file_name
            The file name for the figure.
        tight_layout
            If fig.tight_layout() should be called.
            Does not work with ``plot_path_in_pca``.
        """
        # title = fig.axes[0].get_title()
        # fig.axes[0].set_title(self.plot_title if not title else f'{self.plot_title}\n{title}')
        fig.suptitle(self.plot_title)
        if tight_layout:
            fig.tight_layout()
        for extension in self.extensions:
            fig.savefig(self.path / f"{file_name}.{extension.split('.')[-1]}", dpi=self.dpi)  # remove leading .

    def plot_optimization_progress(self,
                                   file_name: Optional[str] = None,
                                   ratio: Optional[tuple[str, list[str]]] = None,
                                   n_points: int = 100,
                                   **fig_kwargs,
                                   ) -> tuple[plt.Figure, plt.Axes]:
        """Shows the error as function of the iteration number.

        Args
        ----
        file_name
            The file name for the image. This should not include any extension.
            If None (default), it is equal to 'plot_optimization_progress'.
        ratio
            If None (default), only the error ratio will be shown.
            If given, the first element indicates the chemical of interest, and the second element the chemicals it is
            compared to. For example ('A', ['A', 'B']), calculates :math:`A / (A+B)`.
            It will plot the ratio for the last point in each prediction.
        n_points
            The number of iterations for which the ratio will be re-calculated.
            These are uniformly distributed over all possible iterations.
        **fig_kwargs
            Additional keyword arguments that are passed on to plt.subplots().
        """
        if file_name is None:
            file_name = 'plot_optimization_progress'
        self._image_exists(file_name)
        fig, ax = plt.subplots(**fig_kwargs)
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
        self.save_image(fig, file_name)
        return fig, ax

    def plot_grouped_by(self,
                        *args: pd.Series,
                        file_name: Optional[str] = None,
                        group_as: Optional[list[str]] = None,
                        show_remaining: bool = True,
                        xtick_rotation: float = 0,
                        **fig_kwargs
                        ) -> tuple[plt.Figure, np.ndarray[plt.Axes]]:
        """Plots the data in args, and allows easy grouping with respect to their index.

        Args
        ----
        *args
            The data that should be plotted.
            The name of each pd.Series will be used in the legend.
            The index of the first pd.Series object will be used to sort the groups with.
        file_name
            The file name for the image. This should not include any extension.
            If None (default), it is equal to 'plot_grouped_by'.
        group_as
            Group the parameters by a key.
            Each parameter can only be matched with one key exactly.
        show_remaining
            Show the parameter which were not matched by any key in group_as.
        xtick_rotation
            The rotation of the x ticks in degrees.
        **fig_kwargs
            Additional keyword arguments that are passed on to plt.subplots().
        """
        if not args:
            raise ValueError('No data was given to be plotted.')
        x_description = args[0].index

        if file_name is None:
            file_name = 'plot_grouped_by'
        self._image_exists(file_name)

        if group_as is None:
            group_as = ['']

        data = []
        index = []
        for n, arg in enumerate(args):  # contain the same input
            assert set(arg.index) == set(x_description)
            data.append(arg)
            index.append(arg.name if not None else n + 1)
        data = pd.DataFrame(data, index=index, columns=x_description)  # aligns the data

        key_hits = []
        for key in group_as:
            key_hits.append(x_description.str.contains(key))
        key_hits = pd.DataFrame(key_hits, columns=x_description, index=group_as)
        total_hits = key_hits.sum(axis=0)

        if any(total_hits > 1):
            raise ValueError(f'An item was matched by multiple keys.\n{total_hits.index[total_hits > 1]}')

        if show_remaining and any(total_hits == 0):
            key_hits = pd.concat([key_hits, (total_hits == 0).to_frame('other').T])

        fig, axs = plt.subplots(len(key_hits), 1, squeeze=False, **fig_kwargs)
        axs = axs.flatten()
        flx, frx = np.inf, -np.inf  # furthest left, furthest right.
        for ax, (group_key, selected_x) in zip(axs, key_hits.iterrows()):
            ax.set_ylabel(group_key)
            data.loc[:, selected_x].T.plot.bar(ax=ax)

            lx, rx = ax.get_xlim()
            flx = min(flx, lx)
            frx = max(frx, rx)

        # remove legend and set bar plots to all have the same x limits, so they align.
        for n, ax in enumerate(axs):
            if n != 0:
                ax.get_legend().remove()
            ax.set_xlim(flx, frx)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=xtick_rotation, fontsize='small')

        self.save_image(fig, file_name)
        return fig, axs

    def plot_path_in_pca(self,
                         file_name: Optional[str] = None,
                         PC1: int = 0,
                         PC2: int = 1,
                         **fig_kwargs
                         ) -> tuple[plt.Figure, np.ndarray[plt.Axes]]:
        """Plots the path in the dimensionally reduced space (by means of principal component analysis).

        Args
        ----
        file_name
            The file name for the image. This should not include any extension.
            If None (default), it is equal to 'plot_path_in_pca'.
        PC1
            The first principal component that should be plotted.
        PC2
            The second principal component that should be plotted.
        **fig_kwargs
            Additional keyword arguments that are passed on to plt.Figure().

        Returns
        -------
        tuple[plt.Figure, np.ndarray[plt.Axes]]
            The figure, and axes of the plot. Axes[0] is the main plot, axes[1] describes the loadings
            of PC1, whereas axes[2] describes the loadings of PC2.
        """
        if file_name is None:
            file_name = 'plot_path_in_pca'
        self._image_exists(file_name)

        fig = plt.figure(**fig_kwargs)
        gs = fig.add_gridspec(2, 2, width_ratios=(1, 4), height_ratios=(4, 1),
                              left=0.15, right=0.83, bottom=0.15, top=0.83,
                              wspace=0.05, hspace=0.05)

        ax = fig.add_subplot(gs[0, 1])
        pca = PCA().fit(X=self.model.all_x)
        scattered = ax.scatter(self.model.all_x.dot(pca.components_[PC1]),
                               self.model.all_x.dot(pca.components_[PC2]),
                               c=np.arange(len(self.model.all_x)))
        ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
        ax_bbox = ax.get_position(original=True)
        # noinspection PyTypeChecker
        cax = plt.axes([0.85, ax_bbox.ymin, 0.05, ax_bbox.size[1]])
        cbar = plt.colorbar(scattered, cax=cax)
        cbar.set_label("iteration")

        ax_pc0 = fig.add_subplot(gs[1, 1])
        ax_pc1 = fig.add_subplot(gs[0, 0])

        x = np.arange(sum(~self.hide_params))
        ticks = self.model.x_description[~self.hide_params].to_list()

        ax_pc0.bar(x, pca.components_[PC1][~self.hide_params])
        ax_pc1.barh(x, pca.components_[PC2][~self.hide_params])

        ax_pc0.set_xlabel(f"component {PC1}, explained variance {pca.explained_variance_ratio_[PC1]:.2f}")
        ax_pc0.set_xticks(x)
        ax_pc0.set_xticklabels(ticks, rotation=90, fontsize='small')
        ax_pc0.tick_params(left=False)

        ax_pc1.set_ylabel(f"component {PC2}, explained variance {pca.explained_variance_ratio_[PC2]:.2f}")
        ax_pc1.set_yticks(x)
        ax_pc1.set_yticklabels(ticks, fontsize="small")
        ax_pc1.tick_params(bottom=False)

        self.save_image(fig, file_name, tight_layout=False)
        return fig, np.array([ax, ax_pc0, ax_pc1])

    def plot_enantiomer_ratio(self,
                              group_as: list[str],
                              ratio_of: list[str],
                              experimental: pd.DataFrame,
                              prediction: pd.DataFrame,
                              last_N: int = 100,
                              file_name: Optional[str] = None,
                              **fig_kwargs
                              ) -> tuple[plt.Figure, plt.Axes]:
        """Groups the data (experimental or predicted), and calculates the fraction each chemical contributes to the
        total sum. E.g. group_as=['1', '2'] and ratio_of=['A', 'B', 'C'] would look for hits with respect to those keys,
        and one of the calculated data points would be: 1_A / (1_A + 1_B + 1_C)

        Args
        ----
        group_as
            Groups the data by a key.
            Each index can only be matched with one key exactly.
        ratio_of
            Calculate the fraction each chemical contributes to the total sum of all chemicals in this group.
        experimental
            The experimental data of the experiment.
        prediction
            The predicted data. This is not computed within the class to allow the user to specify the indices
            of the prediction in more detail.
        last_N
            The number of points counting from the end will be used to calculate the average ratio with.
        file_name
            The file name for the image. This should not include any extension.
            If None (default), it is equal to 'plot_enantiomer_ratio'.
        **fig_kwargs
            Additional keyword arguments that are passed on to plt.subplots().
        """
        if file_name is None:
            file_name = 'plot_enantiomer_ratio'
        self._image_exists(file_name)

        fig, ax = plt.subplots(**fig_kwargs)

        def analyze_data(data: pd.DataFrame, marker, descr):
            for group_n, group in enumerate(group_as):
                grouped_data = data.loc[:, data.columns.str.contains(group)]
                mask_list = [grouped_data.columns.str.contains(element) for element in ratio_of]
                mask_ratio_of = np.array(mask_list).any(axis=0)

                grouped_data_sum = grouped_data.loc[-last_N:, mask_ratio_of].sum(axis=1)
                for element_n, element in enumerate(ratio_of):
                    element_index = np.nonzero(grouped_data.columns.str.contains(element))[0]
                    if len(element_index) == 0:
                        warnings.warn(f'No matching elements were found for {group} with element {element}, skipping entry.')
                        continue
                    elif len(element_index) == 2:
                        warnings.warn(f'Exactly two matching elements were found for {group} with element {element}.'
                                      f'Taking the shortest hit in the assumption that the other one is the labeled '
                                      f'version. Hits: {grouped_data.columns[element_index]}')
                        if len(grouped_data.columns[element_index[0]]) < len(grouped_data.columns[element_index[1]]):
                            element_index = element_index[0]
                        else:
                            element_index = element_index[1]

                    label = f'{descr}: {element}' if group_n == 0 else None
                    ax.scatter(group_n,
                               grouped_data.iloc[:, element_index].divide(grouped_data_sum, axis=0).mean(),
                               marker=marker,
                               label=label,
                               color=f'C{element_n}',
                               alpha=0.7, s=100)

        analyze_data(experimental, marker='_', descr='exp')
        analyze_data(prediction, marker='|', descr='pred')
        ax.legend(ncol=2)
        ax.set_ylabel('fraction')
        ax.set_xticks(np.arange(len(group_as)))
        ax.set_xticklabels(group_as, fontsize="small")
        ax.set_xlabel('chemical')

        self.save_image(fig, file_name)
        return fig, ax

    def plot_rate_over_time(
            self,
            file_name: Optional[str] = None,
            x_min: Optional[float] = None,
            x_max: Optional[float] = None,
            log_scale: bool = False,
            **fig_kwargs
            ) -> tuple[plt.Figure, plt.Axes]:
        """Plots the parameters, x, as a function of time.

        Args
        ----
        file_name
            The file name for the image. This should not include any extension.
            If None (default), it is equal to 'plot_path_in_pca'.
        x_min
            Plot values lower than the min as the min.
        x_max
            Plot values lwoer than the max as the max.
        log_scale
            If true the data will be plotted on log_scale.
        **fig_kwargs
            Additional keyword arguments that are passed on to plt.subplots().
        """
        if file_name is None:
            file_name = 'plot_rate_over_time'
        self._image_exists(file_name)

        if x_min is None:
            x_min = self.model.all_x.to_numpy().min()

        if x_max is None:
            x_max = self.model.all_x.to_numpy().max()

        if log_scale:
            norm = LogNorm(vmin=x_min, vmax=x_max)
        else:
            norm = Normalize(vmin=x_min, vmax=x_max)

        fig, ax = plt.subplots(**fig_kwargs)
        im = ax.imshow(self.model.all_x.loc[:, ~self.hide_params].T, aspect='auto', label='intensity', norm=norm)
        fig.colorbar(im, label='intensity')
        ax.set_yticks(np.arange(sum(~self.hide_params)))
        ax.set_yticklabels(self.model.x_description[~self.hide_params].tolist())
        ax.set_xlabel('iteration')
        self.save_image(fig, file_name)
        return fig, ax

    def plot_rate_sensitivity(self,
                              x_min: float,
                              x_max: float,
                              file_name: Optional[str] = None,
                              max_error: Optional[float] = None,
                              steps: int = 101,
                              **fig_kwargs
                              ) -> (plt.Figure, plt.Axes):
        """Plot the sensitivity of each parameter, x, to modifications. Only a single parameter is modified at once.

        Args
        ----
        x_min
            The minimum value for x.
        x_max
            The maximum value for x
        file_name
            The file name for the image. This should not include any extension.
            If None (default), it is equal to 'plot_path_in_pca'.
        max_error
            All values larger the maximum value will be plotted as the maximum error.
            If None (default), 3 times the lowest value error will be used.
        steps
            The number of different values that will be modeled for each parameter.
        **fig_kwargs
            Additional keyword arguments that are passed on to plt.subplots().
        """
        if file_name is None:
            file_name = 'plot_rate_sensitivity'
        self._image_exists(file_name)

        from mpl_toolkits.axes_grid1 import make_axes_locatable

        x_value = np.geomspace(x_min, x_max, steps)

        ticks = self.model.optimal_x[~self.hide_params]
        errors = np.full((len(x_value), len(ticks)), np.nan)

        # loop over all non-constant values and adjust those
        for col, key in enumerate(tqdm(self.model.optimal_x[~self.hide_params].keys())):
            for row, adjusted_x in enumerate(x_value):
                # insert all values into the plot
                best_X = self.model.optimal_x.copy()
                best_X[key] = adjusted_x
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        prediction = self.RCO.create_prediction(
                            x=best_X.to_numpy(), x_description=self.model.x_description)
                        unweighed_error = self.RCO.calculate_errors(prediction)
                        found_error = self.RCO.calculate_total_error(unweighed_error)
                except InvalidPredictionError:
                    found_error = np.nan
                errors[row, col] = found_error

        fig, ax = plt.subplots(**fig_kwargs)
        if max_error is None:
            max_error = np.nanmin(errors) * 3

        ax.set_yscale('log')
        im = ax.pcolormesh(np.arange(len(ticks)+1), np.geomspace(x_min, x_max, steps+1), errors,
                           norm=Normalize(vmax=max_error), shading='auto', cmap='viridis_r')

        ax.set_xticks(0.5+np.arange(len(ticks)))  # center the ticks
        ax.set_xticklabels(ticks.index, fontsize="small")
        ax.tick_params(axis='x', rotation=45)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", "5%", pad="3%")
        fig.colorbar(im, cax=cax, label="error")
        self.save_image(fig, file_name)
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
            pred = rco.create_prediction(self.complete_optimal_X[run_index].values,
                                         self.complete_optimal_X[run_index].index)
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

    def show_rate_constants(self, max_error: float, index_constant_values: np.ndarray = None) -> tuple[
        plt.Figure, plt.Axes]:
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

        data = pd.DataFrame(data).reset_index(
            drop=True)  # by converting to df first, we assure that the columns are aligned
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
            ax.plot([0, loading1 * x_factor], [0, loading2 * y_factor], color='tab:gray')
            ax.text(loading1 * x_factor, loading2 * y_factor, rate, ha='center', va='bottom')

        ax.set_title('biplot')
        ax.set_xlabel(f'PC {pc1}, explained variance {pca.explained_variance_ratio_[pc1]:.2f}')
        ax.set_ylabel(f'PC {pc2}, explained variance {pca.explained_variance_ratio_[pc2]:.2f}')

        return fig, ax
