from __future__ import annotations

import inspect
import pathlib
import warnings
from collections.abc import Iterable
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm, Normalize
from sklearn.decomposition import PCA
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from delayed_reactant_labeling.optimize import RateConstantOptimizerTemplate, OptimizedModel, OptimizedMultipleModels
from delayed_reactant_labeling.predict import InvalidPredictionError


class VisualizeModel:
    """
    Contains several methods which help creating plots. The methods are split into three main groups:

    #. Visualization of :class:`OptimizedModels<optimize.OptimizedModel>`
    #. Visualization of :class:`OptimizedMultipleModels<optimize.OptimizedMultipleModel>`
    #. Visualization of other data that is provided by the user.

    Each method will save the figure where the name is equal to the function name.

    Parameters
    ----------
    image_path
        The path to the folder where the created images should be stored.
    model
        A single optimized model of which plots should be created.
        If None, and models is not None, the best model in models will be used instead.
        This will issue a warning once.
    models
        Multiple optimized models of which plots should be created. If model is None,
        the best model in models will be used instead.
    rate_constant_optimizer
        The user implemented class of :class:`optimize.RateConstantOptimizerTemplate`.
        Required by some functions.
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
                 image_path: str | pathlib.Path,
                 model: Optional[OptimizedModel] = None,
                 models: Optional[OptimizedMultipleModels] = None,
                 rate_constant_optimizer: Optional[RateConstantOptimizerTemplate] = None,
                 plot_title: Optional[str] = None,
                 hide_params: Optional[np.ndarray] = None,
                 dpi: int = 600,
                 extensions: Optional[Iterable[str] | str] = None,
                 overwrite_image: bool = False):

        self.path = image_path if isinstance(image_path, pathlib.Path) else pathlib.Path(image_path)
        self.path.mkdir(exist_ok=True, parents=True)  # upon image creation we check for duplicates
        self._model = model
        self._models = models
        self._RCO = rate_constant_optimizer
        self.plot_title = plot_title
        self.dpi = dpi
        extensions = [extensions] if isinstance(extensions, str) else extensions  # convert str to list
        self.extensions = ['png', 'svg'] if extensions is None else extensions
        self.overwrite_image = overwrite_image
        self._hide_params = hide_params

    @property
    def model(self):
        if self._model is not None:
            return self._model
        elif self._models is not None:
            warnings.warn('No model was given, using the best model in models instead.')
            self._model = self._models.best
            return self._model
        else:
            raise ValueError(f'Function {inspect.stack()[1].function} requires a model, '
                             f'but no model(s) were given when initializing the class.')

    @property
    def models(self):
        if self._models is None:
            raise ValueError(f'Function {inspect.stack()[1].function} requires multiple models, but None were given.')
        return self._models

    @property
    def RCO(self):
        if self._RCO is None:
            raise ValueError(f'Function {inspect.stack()[1].function} requires a rate_constant_optimizer, '
                             f'but None was given.')
        return self._RCO

    @property
    def hide_params(self):
        if self._hide_params is None:
            self._hide_params = np.zeros(len(self.model.x_description), dtype=bool)
        return self._hide_params

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
            fig.savefig(self.path / f"{file_name}.{extension.split('.')[-1]}",  # remove leading .
                        dpi=self.dpi, bbox_inches='tight')

    def plot_optimization_progress(self, *,
                                   file_name: Optional[str] = None,
                                   ratio: Optional[tuple[str, list[str]]] = None,
                                   n_points: int = 100,
                                   **fig_kwargs,
                                   ) -> tuple[plt.Figure, plt.Axes]:
        """Shows the error as function of the iteration number.

        **requires a single model**

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
                        group_by: Optional[list[str]] = None,
                        show_remaining: bool = True,
                        xtick_rotation: float = 0,
                        **fig_kwargs
                        ) -> tuple[plt.Figure, np.ndarray[plt.Axes]]:
        """Plots the data in args, and allows easy grouping with respect to their index.

        **requires no model**

        Args
        ----
        *args
            The data that should be plotted.
            The name of each pd.Series will be used in the legend.
            The index of the first pd.Series object will be used to sort the groups with.
        file_name
            The file name for the image. This should not include any extension.
            If None (default), it is equal to 'plot_grouped_by'.
        group_by
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

        if group_by is None:
            group_by = ['']

        data = []
        index = []
        for n, arg in enumerate(args):  # contain the same input
            assert set(arg.index) == set(x_description)
            data.append(arg)
            index.append(arg.name if not None else n + 1)
        data = pd.DataFrame(data, index=index, columns=x_description)  # aligns the data

        key_hits = []
        for key in group_by:
            key_hits.append(x_description.str.contains(key))
        key_hits = pd.DataFrame(key_hits, columns=x_description, index=group_by)
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

    def plot_path_in_pca(self, *,
                         file_name: Optional[str] = None,
                         PC1: int = 0,
                         PC2: int = 1,
                         **fig_kwargs
                         ) -> tuple[plt.Figure, np.ndarray[plt.Axes]]:
        """Plots the path in the dimensionally reduced space (by means of principal component analysis).

        **requires a single model**

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
                              left=0.15, right=0.80, bottom=0.15, top=0.8,
                              wspace=0.05, hspace=0.05)

        ax = fig.add_subplot(gs[0, 1])
        pca = PCA().fit(X=self.model.all_x)
        scattered = ax.scatter(self.model.all_x.dot(pca.components_[PC1]),
                               self.model.all_x.dot(pca.components_[PC2]),
                               c=np.arange(len(self.model.all_x)))
        ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
        ax_bbox = ax.get_position(original=True)
        # noinspection PyTypeChecker
        cax = plt.axes([0.82, ax_bbox.ymin, 0.05, ax_bbox.size[1]])
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
                              group_by: list[str],
                              ratio_of: list[str],
                              experimental: pd.DataFrame,
                              prediction: pd.DataFrame, *,
                              last_N: int = 100,
                              file_name: Optional[str] = None,
                              warn_label_assumption: bool = True,
                              **fig_kwargs
                              ) -> tuple[plt.Figure, plt.Axes]:
        """Groups the data (experimental or predicted), and calculates the fraction each chemical contributes to the
        total sum. E.g. group_as=['1', '2'] and ratio_of=['A', 'B', 'C'] would look for hits with respect to those keys,
        and one of the calculated data points would be: 1_A / (1_A + 1_B + 1_C)

        **requires a single model**

        Args
        ----
        group_by
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
        warn_label_assumption
            If true (default), a warning will be issued when exactly two matches are found for each combination of the 
            elements in group_as and ratio_of. This can be caused by e.g. both '3' and 'D' being present in the strings
            "3_bla_D" and "3_bla_D-custom_label". To resolve this tie the shorted string will be used as it is most
            likely to be the non-labeled compound.  
        **fig_kwargs
            Additional keyword arguments that are passed on to plt.subplots().
        """
        if file_name is None:
            file_name = 'plot_enantiomer_ratio'
        self._image_exists(file_name)

        fig, ax = plt.subplots(**fig_kwargs)

        def analyze_data(data: pd.DataFrame, marker, descr):
            for group_n, group in enumerate(group_by):
                grouped_data = data.loc[:, data.columns.str.contains(group)]

                ratio_of_index = []  # index of hits for each element in ratio_of, corrected for multiple matches
                for ratio_of_element in ratio_of:
                    el_ind = np.nonzero(grouped_data.columns.str.contains(ratio_of_element))[0]
                    if len(el_ind) == 0:
                        warnings.warn(f'No matching columns were found for {group} with item {ratio_of_element} '
                                      f'in the {descr} data, skipping entry.')
                        continue
                    elif len(el_ind) == 1:
                        el_ind = el_ind[0]  # from list to int
                    elif len(el_ind) == 2:
                        el0, el1 = grouped_data.columns[el_ind]
                        if warn_label_assumption:
                            warnings.warn(
                                f'Exactly two matching columns were found for {group} with item {ratio_of_element}.'
                                f'Taking the shortest hit in the assumption that the other one is the labeled '
                                f'version. Hits: {el0, el1}')
                        el_ind = el_ind[0] if el0 < el1 else el_ind[1]
                    else:
                        raise ValueError(f'Too many matching columns were found {group} with items {ratio_of_element}.')
                    ratio_of_index.append(el_ind)

                grouped_data_sum = grouped_data.iloc[-last_N:, ratio_of_index].sum(axis=1)
                for element_n, el_ind in enumerate(ratio_of_index):
                    label = f'{descr}: {ratio_of[element_n]}' if group_n == 0 else None
                    ax.scatter(group_n,
                               grouped_data.iloc[-last_N:, el_ind].divide(grouped_data_sum, axis=0).mean(axis=0),
                               marker=marker,
                               label=label,
                               color=f'C{element_n}',
                               alpha=0.7, s=100)

        analyze_data(experimental, marker='_', descr='exp')
        analyze_data(prediction, marker='|', descr='pred')
        ax.legend(ncol=2)
        ax.set_ylabel('fraction')
        ax.set_xticks(np.arange(len(group_by)))
        ax.set_xticklabels(group_by, fontsize="small")
        ax.set_xlabel('chemical')

        self.save_image(fig, file_name)
        return fig, ax

    def plot_rate_over_time(
            self, *,
            file_name: Optional[str] = None,
            x_min: Optional[float] = None,
            x_max: Optional[float] = None,
            log_scale: bool = False,
            **fig_kwargs
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plots the parameters, x, as a function of time.

        **requires a single model**

        Args
        ----
        file_name
            The file name for the image. This should not include any extension.
            If None (default), it is equal to 'plot_rate_over_time'.
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
                              x_max: float, *,
                              file_name: Optional[str] = None,
                              max_error: Optional[float] = None,
                              steps: int = 101,
                              **fig_kwargs
                              ) -> tuple[plt.Figure, plt.Axes]:
        """Plot the sensitivity of each parameter, x, to modifications. Only a single parameter is modified at once.

        **requires a single model**

        Args
        ----
        x_min
            The minimum value for x.
        x_max
            The maximum value for x
        file_name
            The file name for the image. This should not include any extension.
            If None (default), it is equal to 'plot_rate_sensitivity'.
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
        im = ax.pcolormesh(np.arange(len(ticks) + 1), np.geomspace(x_min, x_max, steps + 1), errors,
                           norm=Normalize(vmax=max_error), shading='auto', cmap='viridis_r')

        ax.set_xticks(0.5 + np.arange(len(ticks)))  # center the ticks
        ax.set_xticklabels(ticks.index, fontsize="small")
        ax.tick_params(axis='x', rotation=45)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", "5%", pad="3%")
        ax.set_ylabel('parameter intensity')
        fig.colorbar(im, cax=cax, label="error")
        self.save_image(fig, file_name)
        return fig, ax

    def plot_error_all_runs(self,
                            top_n: Optional[int] = None, *,
                            file_name: Optional[str] = None,
                            **fig_kwargs
                            ) -> tuple[plt.Figure, plt.Axes]:
        """Plots the error of each model.

        **requires multiple model**

        Args
        ----
        top_n
            How many of the best runs should be plotted.
            If None (default), all runs will be plotted.
        file_name
            The file name for the image. This should not include any extension.
            If None (default), it is equal to 'plot_error_all_runs'.
        **fig_kwargs
            Additional keyword arguments that are passed on to plt.subplots().
        """
        if file_name is None:
            file_name = 'plot_error_all_runs'
        self._image_exists(file_name)

        fig, ax = plt.subplots(**fig_kwargs)
        if top_n is None:
            top_n = self.models.all_optimal_error.shape[0]
        ind = np.argsort(self.models.all_optimal_error)[:top_n]
        ax.scatter(np.arange(ind.shape[0]), self.models.all_optimal_error[ind])
        ax.set_xlabel('run number (sorted by error)')
        ax.set_ylabel('error')
        self.save_image(fig, file_name)
        return fig, ax

    def plot_ratio_all_runs(self,
                            ratio: tuple[str, list[str]], *,
                            top_n: Optional[int] = 20,
                            file_name: Optional[str] = None,
                            **fig_kwargs) -> tuple[plt.Figure, plt.Axes]:
        """Plots the expected ratio with respect to each model, which are sorted on their performance.

        **requires multiple model**

        Args
        ----
        ratio
             The first element indicates the chemical of interest,
             and the second element the chemicals it is compared to.
             For example (‘A’, [‘A’, ‘B’]), calculates $$A/(A+B)$$.
             It will plot the ratio for the last time point in each model.
        top_n
            How many of the best runs should be plotted.
            If None (default), all runs will be plotted.
        file_name
            The file name for the image. This should not include any extension.
            If None (default), it is equal to 'plot_ratio_all_runs'.
        **fig_kwargs
            Additional keyword arguments that are passed on to plt.subplots().
        """
        if file_name is None:
            file_name = 'plot_ratio_all_runs'
        self._image_exists(file_name)

        fig, ax = plt.subplots(**fig_kwargs)

        ind = np.argsort(self.models.all_optimal_error)[:top_n]

        found_ratio = []
        for run_index in ind:
            pred = self.RCO.create_prediction(self.models.all_optimal_x.iloc[run_index, :].values,
                                              self.models.all_optimal_x.iloc[run_index, :].index)
            found_ratio.append(
                (pred[ratio[0]] / pred[ratio[1]].sum(axis=1)).iloc[-1]
            )

        im = ax.scatter(np.arange(len(ind)),
                        found_ratio,
                        c=self.models.all_optimal_error[ind], )
        fig.colorbar(im, ax=ax, label='error')
        ax.set_xlabel('run number (sorted by error)')
        ax.set_ylabel('ratio')

        self.save_image(fig, file_name)
        return fig, ax

    def plot_x_all_runs(self, index: slice, *,
                        file_name: Optional[str] = None,
                        **fig_kwargs
                        ) -> tuple[plt.Figure, plt.Axes]:
        """Plots a boxplot of the parameters for the selected runs.

        **requires multiple model**

        Args
        ----
        index
            The indices of the runs for which the parameters should be plotted.
            index=slice(1, 5) would plot the results of runs 1 upto and including 4,
            but the best run (0) would be skipped.
            index=slice(-5, None) would plot the results for the 5 worst models.
        file_name
            The file name for the image. This should not include any extension.
            If None (default), it is equal to 'plot_x_all_runs'.
        **fig_kwargs
            Additional keyword arguments that are passed on to plt.subplots().
        """
        if file_name is None:
            file_name = 'plot_x_all_runs'
        self._image_exists(file_name)

        df = self.models.all_optimal_x.iloc[self.models.all_optimal_error.argsort()[index], ~self.hide_params]
        fig, ax = plt.subplots(**fig_kwargs)
        ax.boxplot(df.T)
        ax.set_title("distribution of optimal rates")
        ax.set_xticks(1 + np.arange(len(df.columns)))
        ax.set_xticklabels(df.columns, rotation=90)
        ax.set_yscale("log")
        ax.set_ylabel("parameter intensity")
        self.save_image(fig, file_name)
        return fig, ax

    def plot_biplot_all_runs(self,
                             index: slice, *,
                             file_name: Optional[str] = None,
                             PC1: int = 0,
                             PC2: int = 1,
                             **fig_kwargs
                             ) -> tuple[plt.Figure, plt.Axes]:
        """Plots a biplot of the initial and optimal parameters per run.
        This is a dimensionally reduced PCA plot, where the loadings and scores are plotted simultaneously.
        The data is autoscaled (mean=0, std=1) before any analysis, to ensure that its not the scale of a parameter,
        but its deviations which impact the plot.

        **requires multiple model**

        Args
        ----
        index
            The indices of the runs, when sorted by error, for which the parameters should be plotted.
            index=slice(1, 5) would plot the results of runs 1 upto and including 4,
            but the best run (0) would be skipped.
            index=slice(-5, None) would plot the results for the 5 worst models.
        PC1, PC2
            The principal components that should be plotted against each other.
        file_name
            The file name for the image. This should not include any extension.
            If None (default), it is equal to 'plot_biplot_all_runs'.
        **fig_kwargs
            Additional keyword arguments that are passed on to plt.subplots().
        """
        if file_name is None:
            file_name = 'plot_biplot_all_runs'
        self._image_exists(file_name)

        index = self.models.all_optimal_error.argsort()[index]

        data_initial = self.models.all_initial_x.iloc[index, ~self.hide_params].to_numpy()
        data_optimal = self.models.all_optimal_x.iloc[index, ~self.hide_params].to_numpy()
        data = np.concatenate([data_initial, data_optimal], axis=0)

        scaler = StandardScaler()
        pca = PCA().fit(X=scaler.fit_transform(data))

        fig, ax = plt.subplots(**fig_kwargs)

        # scores
        im = ax.scatter(
            scaler.transform(data_initial).dot(pca.components_[PC1]),
            scaler.transform(data_initial).dot(pca.components_[PC2]),
            marker='.', c=self.models.all_optimal_error[index]
        )
        ax.scatter(
            scaler.transform(data_optimal).dot(pca.components_[PC1]),
            scaler.transform(data_optimal).dot(pca.components_[PC2]),
            marker='*', c=self.models.all_optimal_error[index]
        )

        # legend
        ax.scatter(np.nan, np.nan, color="k", marker=".", label="initial")
        ax.scatter(np.nan, np.nan, color="k", marker="*", label="optimal")
        ax.legend()

        # maximize the size of the loadings
        x_factor = abs(np.array(ax.get_xlim())).min() / pca.components_[PC1].max()
        y_factor = abs(np.array(ax.get_ylim())).min() / pca.components_[PC2].max()

        # loadings
        for rate, loading1, loading2 in zip(self.models.x_description[~self.hide_params], pca.components_[PC1], pca.components_[PC2]):
            ax.plot([0, loading1 * x_factor], [0, loading2 * y_factor], color='tab:gray')
            ax.text(loading1 * x_factor, loading2 * y_factor, rate, ha='center', va='bottom')

        ax.set_xlabel(f'PC {PC1}, explained variance {pca.explained_variance_ratio_[PC1]:.2f}')
        ax.set_ylabel(f'PC {PC2}, explained variance {pca.explained_variance_ratio_[PC2]:.2f}')

        fig.colorbar(im, label='error')

        self.save_image(fig, file_name)
        return fig, ax
