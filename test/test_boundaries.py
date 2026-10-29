import ast
from pathlib import Path

import uflx_compiler
import uflx_oracle
import uflx_spec_interpreter
import uflx_toy


def imported(path):
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            yield from (a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            yield node.module


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
