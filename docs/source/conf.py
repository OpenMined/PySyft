# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# stdlib
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath("../../packages"))

# -- Project information -----------------------------------------------------

project = "PySyft"
copyright = "2021, Openmined Community"
author = "Openmined Community"

# https://github.com/sphinx-doc/sphinx/pull/2325/files
# Workaround for sphinx-build recursion limit overflow:
# pickle.dump(doctree, f, pickle.HIGHEST_PROTOCOL)
#  RuntimeError: maximum recursion depth exceeded while pickling an object
#
# Python's default allowed recursion depth is 1000.
sys.setrecursionlimit(5000)

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
# sphinxext.

extensions = [
    "sphinx.ext.autodoc",  # Core library for html generation from docstrings
    "sphinx.ext.autosummary",  # Create neat summary tables
    # "sphinx.ext.doctest",
    "sphinx.ext.extlinks",  # Shorten external links
    # "sphinx.ext.todo",
    "sphinx_copybutton",
    "sphinx.ext.githubpages",
    "sphinx.ext.napoleon",  # handle NumPy documentation formatted docstrings
    "IPython.sphinxext.ipython_directive",  # Sphinx directive to support embedded IPython code
    "IPython.sphinxext.ipython_console_highlighting",
    "sphinx.ext.coverage",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_panels",
]

exclude_patterns = [
    "**.ipynb_checkpoints",
    # to ensure that include files (partial pages) aren't built, exclude them
    # https://github.com/sphinx-doc/sphinx/issues/1965#issuecomment-124732907
    "**/includes/**",
]


autodoc_member_order = "bysource"

autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# Numpydoc
numpydoc_attributes_as_param_list = False

# sphinx-panels shouldn't add bootstrap css since the pydata-sphinx-theme
# already loads it
panels_add_bootstrap_css = False

# Syntax highlight options
pygments_style = "sphinx"

# -- Napoleon Settings -------------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = False
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

autosummary_generate = True
autodoc_typehints = "none"
autoapi_dirs = ["../../packages/syft/src/syft"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".

html_static_path = ["_static"]
html_logo = "_static/logo.png"
html_favicon = "_static/favicon.ico"

html_css_files = [
    "css/pysyft.css",
]


html_theme_options = {
    "external_links": [],
    "github_url": "https://github.com/OpenMined/PySyft",
    "twitter_url": "https://twitter.com/openminedorg",
    "navbar_end": ["navbar-icon-links"],
}
