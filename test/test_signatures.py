import inspect
import re
from pathlib import Path

import pytest

import uflx_spec_interpreter

SPEC = Path(__file__).parents[2] / "uflx" / "design" / "LANGUAGE.md"


def spec_signatures():
    # The free functions of the table rows of the specification, with their parameter names.
    rows = re.findall(r"(?:^\| |\[)`([a-z_]+)\(([^`)]*)\)`", SPEC.read_text(), re.M)
    return {name: [a.strip() for a in args.split(",") if a.strip()] for name, args in rows}


@pytest.mark.skipif(not SPEC.exists(), reason="the uflx repository is not beside this one")
def test_the_public_free_functions_take_the_parameters_of_the_specification():
    spec = spec_signatures()
    # side_map takes the side tables until the core has topologies.
    stand_ins = {"num", "side_map"}
    for name in uflx_spec_interpreter.__all__:
        if name in stand_ins:
            continue
        if inspect.isclass(getattr(uflx_spec_interpreter, name)):
            continue
        params = list(inspect.signature(getattr(uflx_spec_interpreter, name)).parameters)
        assert params == spec[name], name
