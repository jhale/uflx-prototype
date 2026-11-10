"""The core and optimisation policy remain importable without the optional backend."""

import os
import subprocess
import sys


def test_missing_backend_is_lazy_and_actionable():
    code = """
import importlib.abc
import sys

class NoEgglog(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "egglog" or fullname.startswith("egglog."):
            raise ModuleNotFoundError("blocked for optional-dependency test", name="egglog")

sys.meta_path.insert(0, NoEgglog())
from uflx_spec_interpreter import num
from uflx_spec_interpreter._src.optimisation import OptimisationOptions, optimise
assert not any(name.startswith("egglog") for name in sys.modules)
assert OptimisationOptions().iterations == 3
try:
    optimise(num(1) + num(2))
except ImportError as error:
    assert "optimisation" in str(error) or "Python 3.12" in str(error)
else:
    raise AssertionError("missing backend did not raise")
"""
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=os.environ,
    )
