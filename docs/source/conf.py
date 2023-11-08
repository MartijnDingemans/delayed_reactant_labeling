import sys
sys.path.append('../../src')
sys.path.append('../../src/delayed_reactant_labeling')
import delayed_reactant_labeling
import predict
import optimize

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Delayed-Reactant-Labeling'
# copyright = '2023, Martijn Dingemans'
author = 'Martijn Dingemans'
release = '0.2.5'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx_rtd_theme',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
]
templates_path = ['_templates']
exclude_patterns = []
toc_object_entries_show_parents = 'hide'

# autodoc settings
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'

# Napoleon settings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_ivar = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']
html_css_files = ['custom.css']

