import ctypes
import shutil
import subprocess

import basix
import numpy as np
import pytest
from testutil import push_forward

from uflx_compiler import compile_form, evaluate
from uflx_compiler.c_code import c_kernel
from uflx_spec_interpreter import RD, divergence, function, gradient, image, integral, tensor
from uflx_toy.basix_cells import basix_cell
from uflx_toy.basix_spaces import basix_space
from uflx_toy.topology import static_topology

P, RT = basix.ElementFamily.P, basix.ElementFamily.RT
T = static_topology((basix_cell(basix.CellType.triangle),), (((0, 1, 2),),), ())
NODES = np.array([(0.1, 0.0), (1.0, 0.2), (0.3, 0.9), (0.7, 0.6), (0.2, 0.45), (0.55, 0.1)])


def read(label, k):
    # The coordinates are the interleaved coordinate_dofs of FFCx, and a coefficient is w.
    match label:
        case "coefficients:x0":
            return f"coordinate_dofs[{3 * k}]"
        case "coefficients:x1":
            return f"coordinate_dofs[{3 * k + 1}]"
    return f"w[{k}]"


def forms():
    X = basix_space("X", T, P, 2)
    D = image(tensor((function(X, "x0"), function(X, "x1"))), RD(2))
    v = push_forward(function(basix_space("R", T, RT, 1), "v"), D)
    g = gradient(push_forward(function(basix_space("V", T, P, 2), "u"), D), D)
    return {
        "energy": (integral(g[0] * g[0] + g[1] * g[1], D), 6),
        "divergence": (integral(divergence(v, D) * divergence(v, D), D), 3),
    }


@pytest.mark.skipif(shutil.which("cc") is None, reason="no C compiler")
def test_the_c_kernels_give_the_values_of_the_evaluator(tmp_path):
    rng = np.random.default_rng(0)
    xyz = np.zeros((6, 3))
    xyz[:, :2] = NODES
    for name, (f, n) in forms().items():
        compiled = compile_form(f, 4)
        source, library = tmp_path / f"{name}.c", tmp_path / f"{name}.so"
        source.write_text("#include <math.h>\n" + c_kernel(compiled, name, read))
        subprocess.run(["cc", "-O2", "-shared", "-fPIC", source, "-o", library], check=True)
        w = rng.normal(size=n)
        A = np.zeros(1)
        pointer = ctypes.POINTER(ctypes.c_double)
        kernel = getattr(ctypes.CDLL(str(library)), name)
        kernel(
            *(a.ctypes.data_as(pointer) for a in (A, w)),
            None,
            xyz.ctypes.data_as(pointer),
            None,
            None,
            None,
        )
        label = "coefficients:v" if name == "divergence" else "coefficients:u"
        data = {"coefficients:x0": NODES[:, 0], "coefficients:x1": NODES[:, 1], label: w}
        assert np.isclose(A[0], evaluate(compiled, data), rtol=1e-12)
