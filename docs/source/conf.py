import os
import sys

# Ensure path to project root is correct
sys.path.insert(0, os.path.abspath("../.."))

project = "SNPio"
copyright = "2023, Bradley T. Martin and Tyler K. Chafin"
author = "Bradley T. Martin and Tyler K. Chafin"
release = "1.1.0"

# Sphinx extensions for documentation
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",  # Ensure this package is installed
    "sphinx.ext.todo",
]

# Include todos in the documentation
todo_include_todos = True

# Paths for templates
templates_path = ["_templates"]

# Files or directories to ignore during documentation
exclude_patterns = ["**/setup.rst", "**/tests.rst"]

# HTML output theme
html_theme = "sphinx_rtd_theme"

# Optional GitHub URL (not supported natively by sphinx_rtd_theme)
# You may want to integrate this using a custom extension
github_url = "https://github.com/btmartin721/SNPio"
