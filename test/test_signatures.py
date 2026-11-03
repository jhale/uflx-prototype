import inspect
import os
import re
from pathlib import Path

import pytest

import uflx_spec_interpreter

SPEC = Path(
    os.environ.get("UFLX_SPEC_PATH", Path(__file__).parents[2] / "uflx" / "design" / "language.md")
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
    assert set(spec) <= set(uflx_spec_interpreter.__all__)
    stand_ins = {"num", "replace"}
    for name in uflx_spec_interpreter.__all__:
        if name in stand_ins:
            continue
        if inspect.isclass(getattr(uflx_spec_interpreter, name)):
            continue
        signature = inspect.signature(getattr(uflx_spec_interpreter, name))
        params = tuple(signature.parameters)
        assert params == max(spec[name], key=len), name
        for overload in spec[name]:
            assert params[: len(overload)] == overload, name
            signature.bind(**dict.fromkeys(overload))
