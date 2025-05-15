import os
import sys

# -- Project information -----------------------------------------------------
project = "SNPio"
copyright = "2023, Bradley T. Martin and Tyler K. Chafin"
author = "Drs. Bradley T. Martin and Tyler K. Chafin"
release = "1.3.2"

# -- Path setup --------------------------------------------------------------
# Add the project's root directory to sys.path
sys.path.insert(0, os.path.abspath("../../../"))

# -- Sphinx Extensions -------------------------------------------------------
# Add extensions for autodoc, type hints, and more
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # Supports Google-style docstrings
    "sphinx_autodoc_typehints",  # Type hints in function signatures
    "sphinx.ext.todo",  # To-do directives in documentation
    "sphinx.ext.viewcode",  # Add links to highlighted source code
]

# Enable displaying todos
todo_include_todos = True

# -- HTML output theme and customization -------------------------------------
html_theme = "sphinx_rtd_theme"  # Read the Docs theme

html_context = {
    "display_github": True,  # Enable GitHub integration
    "github_user": "btmartin721",  # GitHub username
    "github_repo": "SNPio",  # GitHub repo
    "github_version": "master",  # Branch to use
    "conf_py_path": "/docs/source/",  # Path to docs in the repo
    "current_version": "v1.2.2",  # Project version
    "display_version": True,  # Display version number in the theme
    "latest_version": "master",  # Define the latest stable version
    "display_edit_on_github": True,  # Add 'Edit on GitHub' link
}

# Set paths for templates and static files (custom CSS)
templates_path = ["_templates"]
html_static_path = ["_static"]

# Custom logo and favicon
html_logo = "../../../snpio/img/snpio_logo.png"


# Add custom CSS for further styling if needed
def setup(app):
    app.add_css_file("custom.css")  # Use a custom CSS file (if needed)


# -- General configuration ---------------------------------------------------
# Files or directories to ignore during build
exclude_patterns = ["**/setup.rst", "**/tests.rst", "_build", "Thumbs.db", ".DS_Store"]
