import ast
from pathlib import Path

import uflx_spec_interpreter
import uflx_toy

# Imports go from the oracle to the toy extension to the core, never back.
FORBIDDEN = {uflx_spec_interpreter: {"uflx_toy", "uflx_oracle"}, uflx_toy: {"uflx_oracle"}}


def imported(path):
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            yield from (a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            yield node.module.split(".")[0]


def test_the_core_imports_neither_the_extension_nor_the_oracle():
    for package, forbidden in FORBIDDEN.items():
        for path in Path(package.__file__).parent.rglob("*.py"):
            assert not forbidden & set(imported(path)), path
