# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT

import ast
import subprocess
import sys
from importlib.util import resolve_name
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[2] / "src"
PUBLIC_MODULES = {
    f"uflx.{path.stem}" for path in (SOURCE / "uflx").glob("*.py") if not path.stem.startswith("_")
}


DEPENDENCIES = {
    "uflx": set(),
    "uflx_basix": {"uflx"},
    "uflx_extensions": {"uflx"},
    "uflx_jax_compiler": {"uflx", "uflx_basix", "uflx_extensions"},
}


def imports_in(source, module, is_package=False, root=SOURCE):
    """Resolve static imports, including local and TYPE_CHECKING imports."""
    package = module if is_package else module.rpartition(".")[0]
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            yield from (a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            origin = (
                resolve_name("." * node.level + (node.module or ""), package)
                if node.level
                else node.module
            )
            for name in node.names:
                candidate = f"{origin}.{name.name}"
                path = root.joinpath(*candidate.split("."))
                # `from package import submodule` imports the submodule, while
                # a function or class import belongs to its declaring module.
                yield candidate if path.is_dir() or path.with_suffix(".py").is_file() else origin


def imported(path, root=SOURCE):
    relative = path.relative_to(root).with_suffix("")
    parts = relative.parts[:-1] if path.name == "__init__.py" else relative.parts
    yield from imports_in(path.read_text(), ".".join(parts), path.name == "__init__.py", root)


def files(package):
    return (SOURCE / package).rglob("*.py")


def assert_imports_none_of(package, forbidden):
    for path in files(package):
        assert not forbidden & {m.split(".")[0] for m in imported(path)}, path


@pytest.mark.parametrize("package", DEPENDENCIES)
def test_package_dependencies(package):
    forbidden = DEPENDENCIES.keys() - {package} - DEPENDENCIES[package]
    assert_imports_none_of(package, forbidden)


@pytest.mark.parametrize("package", ["uflx_basix", "uflx_extensions"])
def test_extension_uses_public_core_modules(package):
    for path in files(package):
        modules = PUBLIC_MODULES
        if package == "uflx_basix" and path.name != "quadrature.py":
            modules = modules - {"uflx.algorithms"}
        for m in imported(path):
            if m.partition(".")[0] == "uflx":
                assert m in modules, (path, m)


def test_suite_imports_follow_package_order():
    root = SOURCE.parent
    for path in (root / "test").rglob("*.py"):
        folder = path.relative_to(root / "test").parts[0]
        # Helpers directly under test/ are available to every suite.
        owner = folder if folder in DEPENDENCIES else "uflx"
        forbidden = DEPENDENCIES.keys() - {owner} - DEPENDENCIES[owner]
        if owner in {"uflx", "uflx_extensions"}:
            forbidden.add("basix")
        for module in imported(path, root):
            package = module.removeprefix("test.").partition(".")[0]
            assert package not in forbidden, (path.relative_to(root), module)


def test_importing_the_package_root_does_not_load_or_reexport_the_language():
    code = """
import sys
import uflx
assert not any(name.startswith('uflx.') for name in sys.modules)
assert not any(not name.startswith('_') for name in vars(uflx))
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_core_implementation_does_not_import_its_facade_or_draft_extensions():
    for path in files("uflx"):
        relative = path.relative_to(SOURCE / "uflx")
        if relative.parts[0] != "_src":
            continue
        for module in imported(path):
            assert module != "uflx" and module not in PUBLIC_MODULES, (path, module)
            assert not module.startswith("uflx._draft"), (path, module)


@pytest.mark.parametrize(
    ("source", "module", "is_package", "expected"),
    [
        ("import uflx_jax_compiler as oracle", "uflx.example", False, "uflx_jax_compiler"),
        (
            "from uflx import _src",
            "uflx_basix.example",
            False,
            "uflx._src",
        ),
        (
            "from . import _src",
            "uflx.example",
            False,
            "uflx._src",
        ),
        (
            "from .. import _src",
            "uflx._src.example",
            False,
            "uflx._src",
        ),
        (
            "from ._src import nodes",
            "uflx",
            True,
            "uflx._src.nodes",
        ),
        ("from . import function", "uflx.example", False, "uflx"),
        (
            "if TYPE_CHECKING:\n    from uflx_extensions import external_operators",
            "uflx._src.nodes.example",
            False,
            "uflx_extensions.external_operators",
        ),
        (
            "def f():\n    from uflx._src import arithmetic as a",
            "uflx_basix.example",
            False,
            "uflx._src.arithmetic",
        ),
    ],
)
def test_import_spellings(source, module, is_package, expected):
    assert list(imports_in(source, module, is_package)) == [expected]


def test_private_helpers_stay_in_their_module():
    """Shared interfaces have ordinary names inside the private package."""
    violations = []
    for path in SOURCE.rglob("*.py"):
        module = ".".join(path.relative_to(SOURCE).with_suffix("").parts)
        package = (
            module.removesuffix(".__init__")
            if path.name == "__init__.py"
            else module.rpartition(".")[0]
        )
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.ImportFrom):
                continue
            origin = resolve_name("." * node.level + (node.module or ""), package)
            if not origin.startswith("uflx._src"):
                continue
            violations.extend(
                f"{path.relative_to(SOURCE)}:{node.lineno}: private import {name.name}"
                for name in node.names
                if name.name.startswith("_") and origin != module
            )
    assert not violations, "\n".join(violations)


@pytest.mark.parametrize(
    ("module", "forbidden"),
    [
        ("nodes.expressions", {"algorithms", "rules", "passes"}),
        ("nodes.structure", {"algorithms", "rules", "passes"}),
        (
            "nodes.static",
            {
                "algorithms.rewriting",
                "algorithms.substitution",
                "algorithms.traversal",
                "passes",
            },
        ),
        (
            "algorithms.traversal",
            {"algorithms.rewriting", "algorithms.substitution", "passes"},
        ),
        ("algorithms.rewriting", {"algorithms.substitution", "passes"}),
        ("algorithms.substitution", {"passes"}),
        ("forms", {"passes"}),
        ("rules.structural", {"passes"}),
        ("rules.linearity", {"passes"}),
        ("rules.differentiation", {"passes"}),
        ("rules.coordinates", {"passes"}),
        ("rules.pullback_algebra", {"passes"}),
        ("rules.restriction", {"passes"}),
        ("rules.integration", {"passes"}),
        ("passes.applications", {"passes.contracts", "passes.lowering"}),
        ("passes.derivatives", {"passes.contracts", "passes.lowering"}),
        ("passes.integrals", {"passes.contracts", "passes.lowering"}),
        ("passes.gradients", {"passes.contracts", "passes.lowering"}),
        ("passes.pullbacks", {"passes.contracts", "passes.lowering"}),
        ("passes.spatial", {"passes.contracts", "passes.lowering", "passes.derivatives"}),
        ("passes.contracts", {"passes.lowering"}),
    ],
)
def test_core_dependency_layers(module, forbidden):
    """Queries and transformations must not depend on their callers or recipes."""
    prefix = "uflx._src."
    path = SOURCE.joinpath(*(prefix + module).split(".")).with_suffix(".py")
    for dependency in imported(path):
        assert not any(
            dependency == prefix + name or dependency.startswith(prefix + name + ".")
            for name in forbidden
        ), (module, dependency)
