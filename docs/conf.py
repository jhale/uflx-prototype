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
# Keep the existing example URLs when rendering generated Markdown.
source_suffix = {".rst": "restructuredtext", ".py.md": "markdown", ".md": "markdown"}
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


def convert_examples(app):
    """Render the runnable examples as MyST Markdown before Sphinx discovers its sources."""
    docs = Path(app.srcdir)
    examples = docs.parent / "examples"
    for script in sorted(examples.glob("[!_]*.py")):
        notebook = jupytext.read(script, fmt="py:percent")
        # The license header precedes the first Markdown cell in the script.
        preamble = notebook.cells[0]
        if preamble.cell_type == "code" and all(
            not line or line.startswith("#") for line in preamble.source.splitlines()
        ):
            notebook.cells.pop(0)
        markdown = jupytext.writes(notebook, fmt="md", notebook_metadata_filter="-all")
        markdown += f"\n{{download}}`Download {script.name} <../examples/{script.name}>`\n"
        page = docs / f"{script.name}.md"
        if not page.exists() or page.read_text(encoding="utf-8") != markdown:
            page.write_text(markdown, encoding="utf-8")

    for page in docs.glob("*.py.md"):
        if not (examples / page.stem).is_file():
            page.unlink()


def setup(app):
    app.connect("builder-inited", convert_examples)
    app.connect("autodoc-process-signature", node_signature)
