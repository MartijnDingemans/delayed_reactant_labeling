import pathlib
import shutil

import pandas as pd
import pytest
from pytest import raises

from delayed_reactant_labeling.optimize import OptimizedModel, OptimizedMultipleModels
from delayed_reactant_labeling.visualize import VisualizeModel
from test_optimize import RCO, fake_data

# remove all prior images to make sure that we recreate them!
image_folder = pathlib.Path('./complete_optimization/images')
shutil.rmtree(image_folder, ignore_errors=True)
image_folder.mkdir()

# RCO.optimize_multiple(
#     path='./complete_multiple_optimization/',
#     x_description=x_description,
#     x_bounds=bounds,
#     n_runs=10,
#     n_jobs=-2)
# RCO.optimize(
#     x0=np.array([1, 1, 1]),
#     x_description=x_description,
#     x_bounds=bounds,
#     path='./complete_optimization/')

models = OptimizedMultipleModels('./complete_multiple_optimization')
model = models.best  # OptimizedModel('./complete_optimization')


@pytest.fixture
def VM_fixture():
    return VisualizeModel(
        image_path=image_folder,
        model=model,
        models=models,
        rate_constant_optimizer=RCO,
        plot_title='overwritten!',
        hide_params=None,
        dpi=100,
        extensions='.png',
        overwrite_image=True)


def test_new_method(VM_fixture):
    pass
    # VM_fixture.plot_biplot_all_runs(slice(None))
    # VM_fixture.plot_x_all_runs(slice(5))


def test_extensions():
    path_extensions = pathlib.Path('./complete_optimization/extensions/')
    for extensions in ['.png', 'png', ('png', '.svg'), {'.png', 'svg'}]:
        VSM = VisualizeModel(
            image_path=path_extensions,
            model=model,
            rate_constant_optimizer=RCO,
            plot_title='test_title',
            hide_params=None,
            dpi=100,
            extensions=extensions,
            overwrite_image=True)
        VSM.plot_optimization_progress()
        extensions_list = [extensions] if isinstance(extensions, str) else extensions
        assert len(list(path_extensions.rglob('*'))) == len(extensions_list)
        shutil.rmtree('./complete_optimization/extensions/')


class VSMHelper:
    def __init__(self, VSM: VisualizeModel):
        self.VSM = VSM

    def plot_optimization_progress(self):
        return self.VSM.plot_optimization_progress(ratio=('A-blank', ['A-blank', 'B-blank', 'C-blank']), n_points=10)

    def plot_grouped_by(self):
        # plot x variant
        return self.VSM.plot_grouped_by(
            self.VSM.model.optimal_x.rename('model'),
            pd.Series([0.8, 0.4, 0.67], index=['k-1', 'k1', 'k2'], name='fake1'),
            pd.Series([0.6, 0.3, 0.87], index=['k2', 'k1', 'k-1'], name='fake2'),
            group_by=['k-'], file_name='plot_x', xtick_rotation=90)

    def plot_grouped_by_error(self):
        pred = self.VSM.RCO.create_prediction(self.VSM.model.optimal_x.values,
                                              self.VSM.model.optimal_x.index)
        errors = self.VSM.RCO.calculate_errors(pred)
        weighed_errors = self.VSM.RCO.weigh_errors(errors)
        fig, axs = self.VSM.plot_grouped_by(
            weighed_errors.rename('model'),
            pd.Series([0.01, 0.15, 0.01], index=['ratio_A', 'ratio_C', 'ratio_B'], name='fake1'),
            group_by=['A'], file_name='plot_error')
        self.VSM.save_image(fig, 'plot_error')

    def plot_path_in_pca(self):
        return self.VSM.plot_path_in_pca()

    def plot_enantiomer_ratio(self):
        return self.VSM.plot_enantiomer_ratio(
            group_by=['A', 'B', 'C'],
            ratio_of=['-blank', '-d10'],
            experimental=fake_data,
            prediction=RCO.create_prediction(model.optimal_x.values, model.optimal_x.index)
        )

    def plot_rate_over_time(self):
        return self.VSM.plot_rate_over_time(log_scale=True, x_min=1e-4)

    def plot_rate_sensitivity(self):
        return self.VSM.plot_rate_sensitivity(x_min=1e-6, x_max=1e1, max_error=self.VSM.model.optimal_error*5, steps=11)

    def plot_ratio_all_runs(self):
        return self.VSM.plot_ratio_all_runs(ratio=('A-blank', ['A-blank', 'B-blank', 'C-blank']))

    def plot_error_all_runs(self):
        return self.VSM.plot_error_all_runs()

    def plot_x_all_runs(self):
        return self.VSM.plot_x_all_runs(slice(5))

    def plot_biplot_all_runs(self):
        return self.VSM.plot_biplot_all_runs(slice(6))

    def __iter__(self):
        """iterate over each implemented function"""
        print('\ntesting function:')
        for method in methods_implemented:
            print(method)
            yield getattr(self, method)  # x.method


# outside the function, so we can loop over the implemented methods in different functions
methods_implemented = [method for method in dir(VSMHelper) if method.startswith('_') is False]


def test_all_methods_implemented():
    methods_available = []
    blacklisted_methods = ['save_image', 'RCO', 'model', 'models', 'hide_params']
    for method in dir(VisualizeModel):
        if method.startswith('_') is False and method not in blacklisted_methods:
            methods_available.append(method)

    lacks_n_methods = set(methods_available) - set(methods_implemented)
    assert len(lacks_n_methods) == 0


def test_optimization_progress_empty_folder():
    VSM_overwrite = VisualizeModel(
        image_path=image_folder,
        model=model,
        models=models,
        rate_constant_optimizer=RCO,
        plot_title='empty folder',
        hide_params=None,
        dpi=100,
        extensions='.png',
        overwrite_image=True)

    assert len(list(image_folder.rglob('*.png'))) == 0
    for n, func in enumerate(VSMHelper(VSM_overwrite)):
        # assert that image was saved because the amount of items would have been increased by 1.
        func()
        assert len(list(image_folder.rglob('*.png'))) == n + 1
    assert len(list(image_folder.rglob('*.png'))) == len(list(methods_implemented))


def test_no_accidental_overwrite():
    VSM_no_overwrite = VisualizeModel(
        image_path=image_folder,
        model=model,
        models=models,
        rate_constant_optimizer=RCO,
        plot_title='no overwrite',
        hide_params=None,
        dpi=100,
        extensions='.png',
        overwrite_image=False)

    for n, func in enumerate(VSMHelper(VSM_no_overwrite)):
        with raises(FileExistsError):
            func()


def test_overwriting():
    VSM_overwrite = VisualizeModel(
        image_path=image_folder,
        model=model,
        models=models,
        rate_constant_optimizer=RCO,
        plot_title='',
        hide_params=None,
        dpi=300,
        extensions='.png',
        overwrite_image=True)

    # Before each image will already have been made. check if no additional images are present, or removed.
    L = list(image_folder.rglob('*.png'))
    for n, func in enumerate(VSMHelper(VSM_overwrite)):
        func()
    assert L == list(image_folder.rglob('*.png'))


def test_plot_path_in_pca(VM_fixture):
    VM_fixture.plot_path_in_pca(pc1=1, pc2=2, file_name='pca_usecase1')


def test_missing_parameters():
    no_error = VisualizeModel(
        image_path=image_folder,
        model=model,
        rate_constant_optimizer=RCO,
        plot_title='',
        hide_params=None,
        dpi=300,
        extensions='.png',
        overwrite_image=True)

    no_error.plot_optimization_progress(ratio=('A-blank', ['A-blank', 'B-blank', 'C-blank']))

    no_RCO = VisualizeModel(
        image_path=image_folder,
        model=model,
        # rate_constant_optimizer=RCO,
        plot_title='',
        hide_params=None,
        dpi=300,
        extensions='.png',
        overwrite_image=True)

    with pytest.raises(ValueError, match='Function plot_optimization_progress requires a rate_constant_optimizer'):
        no_RCO.plot_optimization_progress(ratio=('A-blank', ['A-blank', 'B-blank', 'C-blank']))

    no_model = VisualizeModel(
        image_path=image_folder,
        # model=model,
        rate_constant_optimizer=RCO,
        plot_title='',
        hide_params=None,
        dpi=300,
        extensions='.png',
        overwrite_image=True)

    with pytest.raises(ValueError, match='Function plot_optimization_progress requires a model'):
        no_model.plot_optimization_progress(ratio=('A-blank', ['A-blank', 'B-blank', 'C-blank']))

    # different function name
    with pytest.raises(ValueError, match='Function plot_rate_sensitivity requires a model'):
        no_model.plot_rate_sensitivity(x_min=1e-6, x_max=100, steps=5)
