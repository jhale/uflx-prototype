"""Keep tensor-aware arithmetic construction behind its free functions.

This is a source convention check, not shape inference. It recognises imported
constructor names and module aliases, including relative imports. Dynamic
getattr, reassignment of constructors, and wildcard imports are not supported.
"""

import ast
from importlib.util import resolve_name
from pathlib import Path

import pytest

FACTORIES = {
    "Sum": "add",
    "Product": "multiply",
    "Division": "divide",
    "RealPart": "real_part",
    "ImaginaryPart": "imaginary_part",
}
# These functions implement the scalar-to-tensor lifting itself. Calling their
# own free function instead of the scalar constructor would recurse forever.
SCALAR_CONSTRUCTION = {
    ("_src/arithmetic.py", "add", "Sum"),
    ("_src/arithmetic.py", "multiply", "Product"),
    ("_src/arithmetic.py", "divide", "Division"),
}
EXPORT_MODULES = {
    "uflx",
    "uflx.nodes",
    "uflx._src.nodes",
    "uflx._src.nodes.arithmetic",
}


def construction_calls(source, module):
    """Find direct arithmetic constructor calls, resolving import aliases."""
    calls = []

    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self.imports = {}
            self.function = "<module>"

        def resolve(self, expression):
            if isinstance(expression, ast.Name):
                return self.imports.get(expression.id, expression.id)
            if isinstance(expression, ast.Attribute):
                return f"{self.resolve(expression.value)}.{expression.attr}"
            return ""

        def visit_Import(self, node):
            for name in node.names:
                self.imports[name.asname or name.name.split(".")[0]] = (
                    name.name if name.asname else name.name.split(".")[0]
                )

        def visit_ImportFrom(self, node):
            package = module.rpartition(".")[0]
            origin = (
                resolve_name("." * node.level + (node.module or ""), package)
                if node.level
                else node.module
            )
            for name in node.names:
                self.imports[name.asname or name.name] = f"{origin}.{name.name}"

        def visit_FunctionDef(self, node):
            imports, function = self.imports, self.function
            self.imports = imports.copy()
            self.function = node.name
            self.generic_visit(node)
            self.imports, self.function = imports, function

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Call(self, node):
            origin, _, name = self.resolve(node.func).rpartition(".")
            if origin in EXPORT_MODULES and name in FACTORIES:
                calls.append((self.function, name, node.lineno))
            self.generic_visit(node)

    Visitor().visit(ast.parse(source))
    return calls


def test_arithmetic_construction_boundaries():
    root = Path(__file__).resolve().parents[1] / "src" / "uflx"
    violations = []
    exceptions_used = set()
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root).as_posix()
        module = "uflx." + ".".join(path.relative_to(root).with_suffix("").parts)
        for function, node, line in construction_calls(path.read_text(), module):
            key = (relative, function, node)
            if key in SCALAR_CONSTRUCTION:
                exceptions_used.add(key)
            else:
                violations.append(f"{relative}:{line}: use {FACTORIES[node]}(), not {node}()")
    assert not violations, "\n".join(violations)
    assert exceptions_used == SCALAR_CONSTRUCTION, "Remove unused scalar-construction exceptions"


@pytest.mark.parametrize(
    "source",
    [
        "from uflx import Product; Product(a, b)",
        "from uflx._src.nodes import Product as P; P(a, b)",
        "from uflx import nodes; nodes.Product(a, b)",
        "import uflx as u; u.Product(a, b)",
        "import uflx._src.nodes as n; n.Product(a, b)",
        "import uflx._src.nodes; uflx._src.nodes.Product(a, b)",
        "from ..nodes import Product; Product(a, b)",
        "from .. import nodes as n; n.Product(a, b)",
        "from ..nodes.arithmetic import Product as P; P(a, b)",
        "def f():\n    from ..nodes import Product as P\n    return P(a, b)",
    ],
)
def test_constructor_import_spellings(source):
    calls = construction_calls(source, "uflx._src.passes.example")
    assert len(calls) == 1
    assert calls[0][1] == "Product"


def test_patterns_and_free_functions_are_allowed():
    source = """
from uflx import Product, multiply
match expression:
    case Product(a, b):
        result = multiply(a, b)
assert isinstance(result, Product)
"""
    assert construction_calls(source, "uflx._src.example") == []


def test_local_import_alias_does_not_leak():
    source = """
from uflx import multiply as operation
def f():
    from uflx import Product as operation
    return operation(a, b)
def g():
    return operation(a, b)
"""
    calls = construction_calls(source, "uflx._src.example")
    assert [(function, node) for function, node, _ in calls] == [("f", "Product")]


@pytest.mark.parametrize("node", FACTORIES)
def test_each_tensor_aware_operation_is_checked(node):
    source = f"from uflx import {node}; {node}(operand)"
    assert construction_calls(source, "uflx._src.example") == [("<module>", node, 1)]
