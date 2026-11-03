import ctypes
import re
import shutil
import subprocess

import basix
import numpy as np
import pytest
from testutil import push_forward

from uflx_compiler import compile_form, evaluate
from uflx_compiler.c_code import c_kernel
from uflx_spec_interpreter import (
    RD,
    component_tensor,
    divergence,
    function,
    gathered_index,
    gradient,
    image,
    index,
    index_sum,
    integral,
    inverse,
    num,
    tensor,
)
from uflx_toy.basix_cells import basix_cell
from uflx_toy.basix_spaces import basix_space
from uflx_toy.indices import index_table
from uflx_toy.runtime import runtime_array
from uflx_toy.topology import static_topology

P, RT = basix.ElementFamily.P, basix.ElementFamily.RT
T = static_topology((basix_cell(basix.CellType.triangle),), (((0, 1, 2),),), ())
NODES = np.array([(0.1, 0.0), (1.0, 0.2), (0.3, 0.9), (0.7, 0.6), (0.2, 0.45), (0.55, 0.1)])


def read(label, k):
    # The coordinates are the interleaved coordinate_dofs of FFCx, and a coefficient is w.
    match label:
        case "coefficients:x0":
            return f"coordinate_dofs[3 * ({k})]"
        case "coefficients:x1":
            return f"coordinate_dofs[3 * ({k}) + 1]"
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


def loop_examples():
    i, j = index("i"), index("j")
    a = runtime_array("a", (2, 3))
    x = a[i, j] * a[i, j]
    yield "nested", index_sum(x, (i, j)), np.arange(6.0).reshape(2, 3)
    rows = index_sum(x, (j,))
    yield "nested_reductions", index_sum(rows * rows, (i,)), np.arange(6.0).reshape(2, 3)
    yield "sibling_scopes", index_sum(x, (i, j)) + index_sum(num(2) * x, (j, i)), np.ones((2, 3))
    transposed = component_tensor(x, (j, i))
    yield "components", index_sum(transposed[j, i], (i, j)), np.arange(6.0).reshape(2, 3)
    weights = tensor((tensor((num(1), num(2), num(3))), tensor((num(4), num(5), num(6)))))
    yield "constant_table", index_sum(a[i, j] * weights[i, j], (i, j)), np.ones((2, 3))
    g = gathered_index(index_table(((2, 0), (1, 2)), 3), (i, j))
    v = runtime_array("a", (3,))
    yield "gather", index_sum(v[g], (i, j)), np.array([2.0, 3.0, 5.0])
    m = runtime_array("a", (2, 2))
    yield "inverse", index_sum(inverse(m)[i, j], (i, j)), np.array([[2.0, 1.0], [3.0, 4.0]])
    v = runtime_array("a", (2,))
    choices = tensor((v[0] * v[0], v[1] + num(3)))
    yield "tensor_selection", index_sum(choices[i], (i,)), np.array([2.0, 4.0])
    empty = runtime_array("a", (0,))
    yield "empty", index_sum(empty[i], (i,)), np.empty(0)


@pytest.mark.skipif(shutil.which("cc") is None, reason="no C compiler")
@pytest.mark.parametrize("name, expression, data", list(loop_examples()))
def test_nested_loops_match_array_evaluation(tmp_path, name, expression, data):
    code = c_kernel(expression, name, lambda label, offset: f"w[{offset}]")
    source, library = tmp_path / "kernel.c", tmp_path / "kernel.so"
    source.write_text("#include <math.h>\n" + code)
    subprocess.run(
        ["cc", "-std=c99", "-pedantic-errors", "-O2", "-shared", "-fPIC", source, "-o", library],
        check=True,
    )
    result = np.array([7.0])
    pointer = ctypes.POINTER(ctypes.c_double)
    kernel = getattr(ctypes.CDLL(str(library)), name)
    kernel(
        result.ctypes.data_as(pointer), data.ctypes.data_as(pointer), None, None, None, None, None
    )
    assert np.isclose(result[0], 7.0 + evaluate(expression, {"a": data}))
    if name == "nested":
        assert re.search(r"for \([^\n]+\) \{\n    for ", code)
    if name in ("constant_table", "gather"):
        assert re.search(r"static const (double|int) table\d+\[2\]\[", code)


def test_loop_body_does_not_grow_with_extent():
    i = index("i")

    def source(n):
        a = runtime_array("a", (n,))
        return c_kernel(index_sum(a[i] * a[i], (i,)), "sum_squares", lambda label, k: f"w[{k}]")

    small, large = source(3), source(3000)
    assert small.count("for (") == large.count("for (") == 1
    assert small.replace("< 3;", "< 3000;") == large
