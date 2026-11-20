import ast
from importlib.util import resolve_name
from pathlib import Path

import pytest

import uflx_compiler
import uflx_oracle
import uflx_spec_interpreter
import uflx_toy

SOURCE = Path(__file__).resolve().parents[1] / "src"


def imports_in(source, module, is_package=False):
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
                path = SOURCE.joinpath(*candidate.split("."))
                # `from package import submodule` imports the submodule, while
                # a function or class import belongs to its declaring module.
                yield candidate if path.is_dir() or path.with_suffix(".py").is_file() else origin


def imported(path):
    relative = path.relative_to(SOURCE).with_suffix("")
    parts = relative.parts[:-1] if path.name == "__init__.py" else relative.parts
    yield from imports_in(path.read_text(), ".".join(parts), path.name == "__init__.py")


def files(package):
    return Path(package.__file__).parent.rglob("*.py")


def assert_imports_none_of(package, forbidden):
    for path in files(package):
        assert not forbidden & {m.split(".")[0] for m in imported(path)}, path


def test_the_core_imports_neither_the_extension_nor_the_oracle():
    # Imports go from the oracle and the compiler to the toy extension to the core, never back,
    # and the compiler never reads the oracle.
    assert_imports_none_of(uflx_spec_interpreter, {"uflx_toy", "uflx_oracle", "uflx_compiler"})
    assert_imports_none_of(uflx_toy, {"uflx_oracle", "uflx_compiler"})
    assert_imports_none_of(uflx_compiler, {"uflx_oracle"})


def test_the_extension_reads_only_the_public_api_and_the_node_classes_of_the_core():
    for path in files(uflx_toy):
        for m in imported(path):
            if m.startswith("uflx_spec_interpreter"):
                assert m in ("uflx_spec_interpreter", "uflx_spec_interpreter._src.nodes"), path


def test_the_public_api_is_the_names_it_declares():
    public = {n for n in dir(uflx_spec_interpreter) if not n.startswith("_")}
    assert public == set(uflx_spec_interpreter.__all__)
    assert uflx_oracle.evaluate


def test_core_implementation_does_not_import_its_facade_or_draft_extensions():
    for path in files(uflx_spec_interpreter):
        relative = path.relative_to(SOURCE / "uflx_spec_interpreter")
        if relative.parts[0] != "_src" or "extensions" in relative.parts:
            continue
        for module in imported(path):
            assert module != "uflx_spec_interpreter", (path, module)
            assert not module.startswith(
                (
                    "uflx_spec_interpreter._draft",
                    "uflx_spec_interpreter._src.extensions",
                )
            ), (path, module)


@pytest.mark.parametrize(
    ("source", "module", "is_package", "expected"),
    [
        ("import uflx_oracle as oracle", "uflx_spec_interpreter.example", False, "uflx_oracle"),
        (
            "from uflx_spec_interpreter import _src",
            "uflx_toy.example",
            False,
            "uflx_spec_interpreter._src",
        ),
        (
            "from . import _src",
            "uflx_spec_interpreter.example",
            False,
            "uflx_spec_interpreter._src",
        ),
        (
            "from .. import _src",
            "uflx_spec_interpreter._src.example",
            False,
            "uflx_spec_interpreter._src",
        ),
        (
            "from ._src import nodes",
            "uflx_spec_interpreter",
            True,
            "uflx_spec_interpreter._src.nodes",
        ),
        ("from . import function", "uflx_spec_interpreter.example", False, "uflx_spec_interpreter"),
        (
            "if TYPE_CHECKING:\n    from ..extensions import external_operators",
            "uflx_spec_interpreter._src.nodes.example",
            False,
            "uflx_spec_interpreter._src.extensions.external_operators",
        ),
        (
            "def f():\n    from uflx_spec_interpreter._src import arithmetic as a",
            "uflx_toy.example",
            False,
            "uflx_spec_interpreter._src.arithmetic",
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
            if not origin.startswith("uflx_spec_interpreter._src"):
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
        ("algorithms.form_binding", {"passes"}),
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
    prefix = "uflx_spec_interpreter._src."
    path = SOURCE.joinpath(*(prefix + module).split(".")).with_suffix(".py")
    for dependency in imported(path):
        assert not any(
            dependency == prefix + name or dependency.startswith(prefix + name + ".")
            for name in forbidden
        ), (module, dependency)
