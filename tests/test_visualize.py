import pandas as pd
import pathlib
import shutil

import pandas as pd
import pytest
from pytest import raises

from delayed_reactant_labeling.optimize import OptimizedModel
from delayed_reactant_labeling.visualize import VisualizeSingleModel
from test_optimize import RCO

# remove all prior images to make sure that we recreate them!
image_folder = pathlib.Path('./complete_optimization/images')
shutil.rmtree(image_folder, ignore_errors=True)
image_folder.mkdir()

# RCO.optimize(
#     x0=np.array([1, 1, 1]),
#     x_description=x_description,
#     x_bounds=bounds,
#     path='./complete_optimization/')
model = OptimizedModel('./complete_optimization')


def test_extensions():
    path_extensions = pathlib.Path('./complete_optimization/extensions/')
    for extensions in ['.png', 'png', ('png', '.jpg'), {'.png', 'jpg'}]:
        VSM = VisualizeSingleModel(
            path=path_extensions,
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
    def __init__(self, VSM: VisualizeSingleModel):
        self.VSM = VSM

    def plot_optimization_progress(self):
        return self.VSM.plot_optimization_progress(ratio=('A', ['A', 'B', 'C']), n_points=10)

    def plot_x(self):
        return self.VSM.plot_x(pd.Series([0.8, 0.4, 0.67], index=['k-1', 'k1', 'k2'], name='fake1'),
                               pd.Series([0.6, 0.3, 0.87], index=['k2', 'k1', 'k-1'], name='fake2'),
                               group_as=['k-'], )

    def __iter__(self):
        """yields each implemented function"""
        for method in methods_implemented:
            yield getattr(self, method)


# outside the function, so we can loop over the implemented methods in different functions
methods_implemented = [method for method in dir(VSMHelper) if method.startswith('_') is False]


def test_all_methods_implemented():
    methods_available = [method for method in dir(VisualizeSingleModel) if method.startswith('_') is False]
    assert set(methods_available) == set(methods_implemented)


def test_optimization_progress_empty_folder():
    def test_L_increases(_func):
        print(_func)
        _func()
        assert len(list(image_folder.rglob('*.png'))) == n + 1

    VSM_overwrite = VisualizeSingleModel(
        path=image_folder,
        model=model,
        rate_constant_optimizer=RCO,
        plot_title='empty folder',
        hide_params=None,
        dpi=100,
        extensions='.png',
        overwrite_image=True)

    assert len(list(image_folder.rglob('*.png'))) == 0
    print('\ntesting function:')
    for n, func in enumerate(VSMHelper(VSM_overwrite)):
        # assert that image was saved because the amount of items would have been increased by 1.
        test_L_increases(func)
    assert len(list(image_folder.rglob('*.png'))) == len(list(methods_implemented))


def test_no_accidental_overwrite():
    VSM_no_overwrite = VisualizeSingleModel(
        path=image_folder,
        model=model,
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
    VSM_overwrite = VisualizeSingleModel(
        path=image_folder,
        model=model,
        rate_constant_optimizer=RCO,
        plot_title='overwritten!',
        hide_params=None,
        dpi=100,
        extensions='.png',
        overwrite_image=True)

    # Before each image will already have been made. check if no additional images are present, or removed.
    L = list(image_folder.rglob('*.png'))
    for n, func in enumerate(VSMHelper(VSM_overwrite)):
        func()
    assert L == list(image_folder.rglob('*.png'))


@pytest.fixture
def VMS_fixture():
    return VisualizeSingleModel(
        path=image_folder,
        model=model,
        rate_constant_optimizer=RCO,
        plot_title='overwritten!',
        hide_params=None,
        dpi=100,
        extensions='.png',
        overwrite_image=True)


def test_plot_x(VMS_fixture):
    VMS_fixture.plot_x(pd.Series([0.8, 0.4, 0.67], index=['k-1', 'k1', 'k2'], name='fake1'),
                       pd.Series([0.6, 0.3, 0.87], index=['k2', 'k1', 'k-1'], name='fake2'),
                       group_as=['k-'], file_name='usecase1')

    VMS_fixture.plot_x(pd.Series([0.8, 0.4, 0.67], index=['k-1', 'k1', 'k2'], name='fake1'),
                       pd.Series([0.6, 0.3, 0.87], index=['k2', 'k1', 'k-1'], name='fake2'),
                       group_as=['k1', 'k2'], show_remaining=False, file_name='usecase2')

    VMS_fixture.plot_x(pd.Series([0.8, 0.4, 0.67], index=['k-1', 'k1', 'k2'], name='fake1'),
                       pd.Series([0.6, 0.3, 0.87], index=['k2', 'k1', 'k-1'], name='fake2'),
                       file_name='usecase3')
