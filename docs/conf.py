# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT

project = "UFLx prototype"
copyright = "2026, Jack S. Hale"

extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon"]
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


def setup(app):
    app.connect("autodoc-process-signature", node_signature)
