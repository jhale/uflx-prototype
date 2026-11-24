# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT

from pathlib import Path

import jupytext

project = "UFLx prototype"
copyright = "2026, Jack S. Hale"

extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon", "myst_parser"]
source_suffix = {".rst": "restructuredtext", ".md": "markdown", ".py": "markdown"}
include_patterns = ["index.md", "docs/*.rst", "examples/[!_]*.py"]
exclude_patterns = [".*", "build", "venv", "src", "test"]
myst_enable_extensions = ["html_image"]
html_theme = "pydata_sphinx_theme"
html_theme_options = {"github_url": "https://github.com/jhale/uflx-prototype"}


def node_signature(app, what, name, obj, options, signature, return_annotation):
    """Show the positional construction fields declared by each concrete node."""
    from uflx.nodes import Base

    if what == "class" and isinstance(obj, type) and issubclass(obj, Base):
        if "identifier" not in vars(obj):
            return "", None
        fields = (*obj.argument_names, "/") if obj.argument_names else ()
        return f"({', '.join(fields)})", None
    return None


def render_example(app, docname, source):
    """Give MyST the Markdown representation of a runnable Python example."""
    script = Path(app.env.doc2path(docname))
    if script.suffix != ".py":
        return

    notebook = jupytext.reads(source[0], fmt="py:percent")
    # The license header precedes the first Markdown cell in the script.
    preamble = notebook.cells[0]
    if preamble.cell_type == "code" and all(
        not line or line.startswith("#") for line in preamble.source.splitlines()
    ):
        notebook.cells.pop(0)
    markdown = jupytext.writes(notebook, fmt="md", notebook_metadata_filter="-all")
    source[0] = markdown + f"\n{{download}}`Download {script.name} <{script.name}>`\n"


def setup(app):
    app.connect("source-read", render_example)
    app.connect("autodoc-process-signature", node_signature)
