# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here.
import pathlib
import sys
import os
import py_sim
import shutil
sys.path.insert(0, pathlib.Path(__file__).parents[3].resolve().as_posix())
sys.path.insert(0, os.path.abspath('.'))

# # Copy the readme up
# readme_filename = pathlib.Path(__file__).parents[2].joinpath("README.md").resolve()
# copy_filename = pathlib.Path(__file__).parent.joinpath("README.md")
# print("current file name: ", readme_filename)
# print("new file name: ", copy_filename)
# shutil.copy(src=readme_filename, dst=copy_filename)

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Python Planning Sim'
copyright = '2023, Greg Droge'
author = 'Greg Droge'
release = '23.06-0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.duration',    # Times the creation of the build
              'sphinx.ext.autodoc',     # Allows for autodocumentation
              'sphinx.ext.autosummary', # Allows for creation of the toc
              'myst_parser',            # Allows for use of markdown files (md)
              'sphinx.ext.napoleon',    # Allows for the use of google-style documentation
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']

#html_theme = 'furo'