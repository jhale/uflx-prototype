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
            "if TYPE_CHECKING:\n    from ..extensions import shape",
            "uflx_spec_interpreter._src.nodes.example",
            False,
            "uflx_spec_interpreter._src.extensions.shape",
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


def test_implementation_definitions_are_public_or_underscore_prefixed():
    root = SOURCE / "uflx_spec_interpreter"
    public = set()
    for facade, package in (
        (root / "__init__.py", "uflx_spec_interpreter"),
        (root / "_src/nodes/__init__.py", "uflx_spec_interpreter._src.nodes"),
    ):
        for node in ast.parse(facade.read_text()).body:
            if isinstance(node, ast.ImportFrom):
                module = resolve_name("." * node.level + (node.module or ""), package)
                public.update((module, name.name) for name in node.names)
    violations = []
    for path in (root / "_src").rglob("*.py"):
        module = ".".join(path.relative_to(SOURCE).with_suffix("").parts)
        for node in ast.parse(path.read_text()).body:
            names = []
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                names = [node.name]
            elif isinstance(node, ast.Assign):
                names = [
                    n.id
                    for target in node.targets
                    for n in ast.walk(target)
                    if isinstance(n, ast.Name)
                ]
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                names = [node.target.id]
            for name in names:
                if not name.startswith("_") and (module, name) not in public:
                    violations.append(f"{path.relative_to(SOURCE)}:{node.lineno}: rename {name}")
    assert not violations, "\n".join(violations)


@pytest.mark.parametrize(
    ("module", "forbidden"),
    [
        ("static", {"rewriting", "substitution", "traversal", "passes", "contracts"}),
        ("traversal", {"rewriting", "substitution", "passes", "contracts"}),
        ("rewriting", {"substitution", "passes", "contracts"}),
        ("substitution", {"passes", "contracts"}),
        ("passes.actions", {"contracts", "passes.lowering"}),
        ("passes.derivatives", {"contracts", "passes.lowering"}),
        ("passes.integrals", {"contracts", "passes.lowering"}),
        ("passes.gradients", {"contracts", "passes.lowering"}),
        ("passes.pullbacks", {"contracts", "passes.lowering"}),
        ("contracts", {"passes.lowering"}),
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
