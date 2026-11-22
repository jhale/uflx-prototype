import ast
import inspect
import os
import re
from pathlib import Path
from typing import get_overloads

import pytest

from uflx import algorithms, arithmetic, calculus, forms, geometry, nodes, passes, tensors

LANGUAGE_MODULES = (arithmetic, calculus, forms, geometry, passes, tensors)

SPEC = Path(
    os.environ.get(
        "UFLX_SPEC_PATH", Path(__file__).resolve().parents[3] / "uflx" / "design" / "language.md"
    )
)


def spec_signatures():
    # The free functions of the table rows of the specification, with their parameter names.
    source = SPEC.read_text()
    chapters = re.findall(r"\]\((language/[^)]+\.md)\)", source)
    source += "\n" + "\n".join((SPEC.parent / chapter).read_text() for chapter in chapters)
    table = source.split("<!-- generated: free-functions", 1)[1]
    table = table.split("<!-- end generated: free-functions", 1)[0]
    rows = re.findall(r"\[`([a-z_]+)\(([^`)]*)\)`", table)
    # Pass signatures have their own tables rather than the free-function appendix.
    rows += re.findall(r"^\| `((?:apply_|expand_)[a-z_]+)\(([^`)]*)\)`", source, re.M)
    signatures = {}
    for name, args in rows:
        parameters = tuple(a.strip() for a in args.split(",") if a.strip())
        signatures.setdefault(name, set()).add(parameters)
    return signatures


@pytest.mark.skipif(not SPEC.exists(), reason="the uflx repository is not beside this one")
def test_the_public_free_functions_take_the_parameters_of_the_specification():
    spec = spec_signatures()
    functions = {
        name: getattr(module, name) for module in LANGUAGE_MODULES for name in module.__all__
    }
    assert set(functions) == set(spec) | {"identity", "num", "reference_integral"}
    assert len(functions) == sum(len(module.__all__) for module in LANGUAGE_MODULES)
    for name in spec:
        function = functions[name]
        signatures = [inspect.signature(f) for f in get_overloads(function) or [function]]
        parameters = {
            tuple(
                "*" + p.name if p.kind == inspect.Parameter.VAR_POSITIONAL else p.name
                for p in signature.parameters.values()
            )
            for signature in signatures
        }
        assert parameters == spec[name], name
        for signature in signatures:
            keywords = {
                p.name: None
                for p in signature.parameters.values()
                if p.kind != inspect.Parameter.VAR_POSITIONAL
            }
            signature.bind(**keywords)
            inspect.signature(function).bind(**keywords)


@pytest.mark.skipif(not SPEC.exists(), reason="the uflx repository is not beside this one")
def test_concrete_node_names_and_ordered_arguments_match_the_class_tables():
    chapters = re.findall(r"\]\((language/[^)]+\.md)\)", SPEC.read_text())
    specified = {}
    for chapter in chapters:
        if chapter == "language/extensions.md":
            continue
        source = (SPEC.parent / chapter).read_text()
        # A class table starts with its constructors, a separator, and its base class.
        headings = re.findall(r"^\| ([^|\n]+)\| *\|\n\|[^\n]+\n\| Extends\b", source, re.M)
        for heading in headings:
            for name, args in re.findall(r"`([A-Z]\w+)\(([^)`]*)\)`", heading):
                specified[name] = tuple(a.strip() for a in args.split(",") if a.strip())
    implemented = {
        name: cls.argument_names
        for name, cls in vars(nodes).items()
        if isinstance(cls, type) and getattr(cls, "__final__", False)
    }
    assert implemented == specified


def facade_declarations(source):
    """Read the facade's explicit imports and literal __all__ declaration."""
    tree = ast.parse(source)
    declarations = [
        n.value
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "__all__" for t in n.targets)
    ]
    assert len(declarations) == 1, "the facade needs one literal __all__ declaration"
    exports = ast.literal_eval(declarations[0])
    assert isinstance(exports, list) and all(isinstance(n, str) for n in exports)
    assert len(exports) == len(set(exports)), "duplicate __all__ entries"
    imports = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            assert all(n.name != "*" for n in node.names), "facade imports must be explicit"
            imports.extend(n.asname or n.name for n in node.names)
        elif isinstance(node, ast.Import):
            imports.extend(n.asname or n.name.split(".")[0] for n in node.names)
    public_imports = [n for n in imports if not n.startswith("_")]
    assert len(public_imports) == len(set(public_imports)), "duplicate facade bindings"
    assert set(exports) == set(public_imports), "__all__ must match the explicit public imports"
    return set(exports)


@pytest.mark.parametrize("module", [*LANGUAGE_MODULES, algorithms, nodes])
def test_facade_imports_match_its_declared_exports(module):
    path = Path(module.__file__)
    assert facade_declarations(path.read_text()) == set(module.__all__)


@pytest.mark.parametrize(
    "source",
    [
        "from ._src.forms import form; __all__ = ['form', 'form']",
        "from ._src.forms import form; __all__ = []",
        "from ._src.forms import form; __all__ = ['form', 'function']",
        "from ._src.forms import *; __all__ = ['form']",
        (
            "from ._src.forms import form; from ._src.forms import function as form; "
            "__all__ = ['form']"
        ),
    ],
)
def test_facade_check_rejects_inconsistent_exports(source):
    with pytest.raises(AssertionError):
        facade_declarations(source)
