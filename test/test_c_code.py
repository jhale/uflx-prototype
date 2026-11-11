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
    broadcast_to,
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
    logarithm,
    num,
    square_root,
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
    contraction = index_sum(choices[i], (i,))
    yield "tensor_contraction", contraction * contraction, np.array([2.0, 4.0])
    v = runtime_array("a", (3,))
    product = v[0] * v[1]
    yield "direct_product", product, np.array([2.0, 3.0, 5.0])
    yield "logarithm", index_sum(logarithm(v[i]), (i,)), np.array([0.5, 1.0, 3.0])
    yield "negative_power", index_sum(v[i] ** num(-2), (i,)), np.array([0.5, 1.0, 3.0])
    yield "zero_power", index_sum(v[i] ** num(0), (i,)), np.array([0.5, 1.0, 3.0])
    yield "fractional_power", v[0] ** (num(1) / num(2)), np.array([2.0, 3.0, 5.0])
    yield "variable_power", v[0] ** v[1], np.array([2.0, 0.5, 5.0])
    yield "hoisted_product", index_sum(product * v[i], (i,)), np.array([2.0, 3.0, 5.0])
    yield (
        "shared_accumulated_product",
        product + index_sum(product * v[i], (i,)),
        np.array([2.0, 3.0, 5.0]),
    )
    repeated = broadcast_to(runtime_array("a", (2, 1)), (3, 2, 4))
    k = index("k")
    yield "broadcast", index_sum(repeated[k, i, j], (k, i, j)), np.array([[2.0], [5.0]])
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
    if name == "direct_product":
        assert "A[0] += w[0] * w[1];" in code
        assert "const double t" not in code
    if name in ("hoisted_product", "shared_accumulated_product"):
        declaration = re.search(r"const double (t\d+) = w\[0\] \* w\[1\];", code)
        assert declaration is not None
        temporary = declaration[1]
        assert code.index(declaration[0]) < code.index("for (")
        if name == "shared_accumulated_product":
            assert f"A[0] += {temporary};" in code
    if name == "nested":
        assert re.search(r"A\[0\] \+= w\[[^;]+ \* w\[[^;]+;", code)
    if name == "tensor_contraction":
        assert "switch" not in code
        assert "for (" not in code
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


@pytest.mark.skipif(shutil.which("cc") is None, reason="no C compiler")
@pytest.mark.parametrize("late_weights", [False, True])
def test_scheduling_shares_equal_reductions_and_hoists_invariants(tmp_path, late_weights):
    q, k, i, j = (index(name) for name in ("q", "k", "i", "j"))
    coefficients = runtime_array("coefficients", (2, 4))
    scale = runtime_array("scale", ())
    first = index_sum(coefficients[q, k] * coefficients[q, k], (k,))
    second = index_sum(coefficients[q, k] * coefficients[q, k], (k,))
    assert first == second
    basis = tensor((tensor((num(1), num(2), num(3))), tensor((num(4), num(5), num(6)))))
    expression = index_sum((first * second + scale * scale) * basis[q, i] * basis[q, j], (q,))

    if late_weights:
        expression = index_sum(first * second * basis[q, i] * basis[q, j] * scale * scale, (q,))

    def counted_read(label, offset):
        counter, start = (0, 0) if label == "coefficients" else (1, 8)
        return f"counted_read({counter}, w[{start} + ({offset})])"

    code = c_kernel(expression, "scheduled", counted_read, output_indices=(i, j))
    source, library = tmp_path / "scheduled.c", tmp_path / "scheduled.so"
    source.write_text(
        "int counts[2] = {0, 0};\n"
        "double counted_read(int counter, double value) { counts[counter]++; return value; }\n"
        + code
    )
    # No optimiser: these counts measure the scheduling performed by our emitter.
    subprocess.run(["cc", "-O0", "-shared", "-fPIC", source, "-o", library], check=True)
    lib = ctypes.CDLL(str(library))
    data = np.arange(1.0, 10.0)
    result = np.full((3, 3), 7.0)
    pointer = ctypes.POINTER(ctypes.c_double)
    lib.scheduled(
        result.ctypes.data_as(pointer),
        data.ctypes.data_as(pointer),
        None,
        None,
        None,
        None,
        None,
    )
    B = np.array([[1, 2, 3], [4, 5, 6]])
    factors = np.sum(data[:8].reshape(2, 4) ** 2, axis=1) ** 2
    factors = factors * data[8] ** 2 if late_weights else factors + data[8] ** 2
    np.testing.assert_allclose(result, 7 + B.T @ (factors[:, None] * B))
    # Each squared input is read twice, once per required point/DOF; the
    # structurally equal reduction and all nine matrix entries share the result.
    assert list((ctypes.c_int * 2).in_dll(lib, "counts")) == [16, 2]
    assert code.count("double sum") == 1
    loops = re.findall(r"for \(int ([a-z]+)\d+", code)
    assert loops[0] == "q"
    assert loops[-2:] == ["a", "a"]
    assert loops.count("i") == 1


@pytest.mark.skipif(shutil.which("cc") is None, reason="no C compiler")
def test_scheduling_preserves_conditional_evaluation(tmp_path):
    i = index("choice")
    values = runtime_array("values", (2,))
    expression = tensor((num(1) / values[0], values[1] * values[1]))[i]
    code = c_kernel(
        expression,
        "selected",
        lambda label, offset: f"counted_read({offset}, w[{offset}])",
        index_bindings={i: "entity_local_index[0]"},
    )
    source, library = tmp_path / "selected.c", tmp_path / "selected.so"
    source.write_text(
        "int counts[2] = {0, 0};\n"
        "double counted_read(int index, double value) { counts[index]++; return value; }\n" + code
    )
    subprocess.run(["cc", "-O0", "-shared", "-fPIC", source, "-o", library], check=True)
    lib = ctypes.CDLL(str(library))
    data, result = np.array([0.0, 3.0]), np.zeros(1)
    pointer = ctypes.POINTER(ctypes.c_double)
    choice = (ctypes.c_int * 1)(1)
    lib.selected(
        result.ctypes.data_as(pointer),
        data.ctypes.data_as(pointer),
        None,
        None,
        choice,
        None,
        None,
    )
    assert result[0] == 9
    assert list((ctypes.c_int * 2).in_dll(lib, "counts")) == [0, 2]


def test_empty_output_does_not_evaluate_hoisted_inputs():
    i = index("empty")
    empty, scale = runtime_array("empty", (0,)), runtime_array("scale", ())
    code = c_kernel(
        empty[i] * (scale * scale),
        "empty_output",
        lambda label, offset: f"w[{offset}]",
        output_indices=(i,),
    )
    assert "w[" not in code
    assert "A[" not in code


@pytest.mark.skipif(shutil.which("cc") is None, reason="no C compiler")
@pytest.mark.parametrize("order", [(0, 1, 2), (2, 0, 1)])
@pytest.mark.parametrize("shadow", [False, True])
def test_array_scheduling_reuses_later_indices(tmp_path, order, shadow):
    q, i, j, k = (index(name) for name in ("q", "i", "j", "k"))
    values = runtime_array("values", (2, 3, 2))
    first = values[q, j, k] * values[q, j, k] + num(1)
    second = values[q, j, k] * values[q, j, k] + num(1)
    row = tensor(tuple(num(v) for v in (1, 2, 3, 4)))
    integrand = row[i] * first * second
    if shadow:
        # This j is locally bound, independently of the output's j.
        integrand = integrand * index_sum(values[0, j, 0] * values[0, j, 0] * values[0, j, 0], (j,))
    expression = index_sum(integrand, (q,))
    code = c_kernel(
        expression,
        "scheduled_array",
        lambda label, offset: f"counted_read(w[{offset}])",
        output_indices=tuple((i, j, k)[axis] for axis in order),
    )
    source, library = tmp_path / "array.c", tmp_path / "array.so"
    source.write_text(
        "int reads = 0;\ndouble counted_read(double value) { reads++; return value; }\n" + code
    )
    subprocess.run(["cc", "-O0", "-shared", "-fPIC", source, "-o", library], check=True)
    lib = ctypes.CDLL(str(library))
    data = np.arange(12.0).reshape(2, 3, 2)
    result = np.full(tuple((4, 3, 2)[axis] for axis in order), 7.0)
    pointer = ctypes.POINTER(ctypes.c_double)
    lib.scheduled_array(
        result.ctypes.data_as(pointer),
        data.ctypes.data_as(pointer),
        None,
        None,
        None,
        None,
        None,
    )
    expected = np.arange(1, 5)[:, None, None] * np.sum((data**2 + 1) ** 2, axis=0)
    if shadow:
        expected *= np.sum(data[0, :, 0] ** 3)
    np.testing.assert_allclose(result, 7 + np.transpose(expected, order))
    assert ctypes.c_int.in_dll(lib, "reads").value == (33 if shadow else 24)
    # Neither the quadrature axis nor the independent output axis needs storage.
    assert re.findall(r"double cache\d+(\[[^;]+);", code) == (
        ["[3][2]"] if order == (0, 1, 2) else ["[2][3]"]
    )


@pytest.mark.skipif(shutil.which("cc") is None, reason="no C compiler")
def test_precomputed_arrays_preserve_conditional_evaluation(tmp_path):
    i, j, choice = (index(name) for name in ("i", "j", "choice"))
    values = runtime_array("values", (2, 3))
    row = tensor(tuple(num(v) for v in (1, 2, 3, 4)))
    selected = tensor((num(1) / values[0, j], values[1, j] * values[1, j]))[choice]
    code = c_kernel(
        row[i] * selected,
        "selected_array",
        lambda label, offset: f"counted_read({offset}, w[{offset}])",
        output_indices=(i, j),
        index_bindings={choice: "entity_local_index[0]"},
    )
    source, library = tmp_path / "selected.c", tmp_path / "selected.so"
    source.write_text(
        "int reads[2] = {0, 0};\n"
        "double counted_read(int offset, double value) { reads[offset / 3]++; return value; }\n"
        + code
    )
    subprocess.run(["cc", "-O0", "-shared", "-fPIC", source, "-o", library], check=True)
    lib = ctypes.CDLL(str(library))
    data = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    result = np.zeros((4, 3))
    pointer = ctypes.POINTER(ctypes.c_double)
    selected_side = (ctypes.c_int * 1)(1)
    lib.selected_array(
        result.ctypes.data_as(pointer),
        data.ctypes.data_as(pointer),
        None,
        None,
        selected_side,
        None,
        None,
    )
    np.testing.assert_allclose(result, np.arange(1, 5)[:, None] * data[1] ** 2)
    assert list((ctypes.c_int * 2).in_dll(lib, "reads")) == [0, 6]
    assert re.findall(r"double cache\d+(\[[^;]+);", code) == ["[3]"]


@pytest.mark.skipif(shutil.which("cc") is None, reason="no C compiler")
@pytest.mark.parametrize("quadrature", [False, True])
def test_component_specialisation_skips_zero_blocks(tmp_path, quadrature):
    i, r, j, s = (index(name) for name in ("i", "r", "j", "s"))
    basis = tensor((num(1), num(2), num(4)))
    diagonal = tensor((tensor((num(1), num(0))), tensor((num(0), num(2)))))
    scale = runtime_array("scale", ())
    expression = square_root(scale) * basis[i] * basis[j] * diagonal[r, s]
    if quadrature:
        q = index("q")
        scale = runtime_array("scale", (2,))
        expression = index_sum(square_root(scale[q]) * basis[i] * basis[j] * diagonal[r, s], (q,))
    code = c_kernel(
        expression,
        "blocked",
        lambda label, offset: f"read_scale(w + ({offset}))",
        output_indices=(i, r, j, s),
        unroll_indices=(r, s),
    )
    # Two diagonal blocks are emitted; the two identically-zero blocks do no work.
    assert code.count("A[") == 2
    # Both component updates share the same two basis loops.
    assert len(re.findall(r"for \(int a\d+ =", code)) == 2
    source, library = tmp_path / "blocked.c", tmp_path / "blocked.so"
    source.write_text(
        "#include <math.h>\nint reads = 0;\n"
        "double read_scale(const double* w) { reads++; return w[0]; }\n" + code
    )
    subprocess.run(["cc", "-O0", "-shared", "-fPIC", source, "-o", library], check=True)
    lib = ctypes.CDLL(str(library))
    data = np.array([9.0, 16.0])
    result = np.full((6, 6), 7.0)
    pointer = ctypes.POINTER(ctypes.c_double)
    lib.blocked(
        result.ctypes.data_as(pointer), data.ctypes.data_as(pointer), None, None, None, None, None
    )
    expected = 7 + (7 if quadrature else 3) * np.kron(
        np.outer([1, 2, 4], [1, 2, 4]), np.diag([1, 2])
    )
    np.testing.assert_allclose(result, expected)
    # Kernel-input computations are shared across specialised blocks, even without -O.
    assert ctypes.c_int.in_dll(lib, "reads").value == (2 if quadrature else 1)


@pytest.mark.skipif(shutil.which("cc") is None, reason="no C compiler")
@pytest.mark.parametrize("nonlinear", [False, True])
def test_material_factoring_preserves_rectangular_contractions(tmp_path, nonlinear):
    i, j = index("i"), index("j")
    x = tensor(tuple(num(v) for v in (1, -2, 3)))
    y = tensor(tuple(num(v) for v in (2, 0, -1)))
    u = tensor(tuple(num(v) for v in (3, -1, 2, 4)))
    v = tensor(tuple(num(v) for v in (-2, 1, 0, 5)))
    material = runtime_array("material", (3,))
    lam, mu, weight = (material[k] for k in range(3))
    expression = weight * (mu * (num(2) * x[i] * u[j] + y[i] * v[j]) + lam * x[i] * u[j])
    if nonlinear:
        # Products of two trial-dependent sums must remain valid, without assuming linearity.
        expression = expression * (u[j] + v[j])
    code = c_kernel(expression, "factored", lambda label, off: f"w[{off}]", output_indices=(i, j))
    source, library = tmp_path / "factored.c", tmp_path / "factored.so"
    source.write_text(code)
    subprocess.run(["cc", "-O2", "-shared", "-fPIC", source, "-o", library], check=True)
    lib = ctypes.CDLL(str(library))
    pointer = ctypes.POINTER(ctypes.c_double)
    for data in np.random.default_rng(23).normal(size=(5, 3)):
        result = np.full((3, 4), 7.0)
        lib.factored(
            result.ctypes.data_as(pointer),
            data.ctypes.data_as(pointer),
            None,
            None,
            None,
            None,
            None,
        )
        expected = data[2] * (
            (data[0] + 2 * data[1]) * np.outer([1, -2, 3], [3, -1, 2, 4])
            + data[1] * np.outer([2, 0, -1], [-2, 1, 0, 5])
        )
        if nonlinear:
            expected *= np.array([1, 0, 2, 9])
        np.testing.assert_allclose(result, 7 + expected, rtol=1e-12, atol=1e-12)


@pytest.mark.skipif(shutil.which("cc") is None, reason="no C compiler")
@pytest.mark.parametrize("curved", [False, True])
def test_geometry_contraction_is_prepared_at_its_invariant_scope(tmp_path, curved):
    from uflx_compiler.quadrature import constant

    counts = []
    for nq in (4, 7):
        q, i, j, a, b = (index(name) for name in ("q", "i", "j", "a", "b"))
        rng = np.random.default_rng(23)
        left, right = rng.normal(size=(2, nq, 4)), rng.normal(size=(2, nq, 5))
        weights = rng.uniform(0.1, 1, size=nq)
        geometry = rng.normal(size=(nq, 2, 2) if curved else (2, 2))
        B = runtime_array("geometry", geometry.shape)
        D, E = constant(left), constant(right)
        terms = []
        for k in range(2):
            first = B[q, k, a] if curved else B[k, a]
            second = B[q, k, b] if curved else B[k, b]
            terms.append(index_sum(first * D[a, q, i], (a,)) * index_sum(second * E[b, q, j], (b,)))
        expression = index_sum(constant(weights)[q] * sum(terms, num(0)), (q,))
        code = c_kernel(
            expression,
            "metric",
            lambda label, off: f"geometry_read(w, {off})",
            output_indices=(i, j),
        )
        source, library = tmp_path / f"metric{nq}.c", tmp_path / f"metric{nq}.so"
        source.write_text(
            "int reads = 0;\n"
            "double geometry_read(const double* w, int k) { reads++; return w[k]; }\n" + code
        )
        # Disable C optimisation to measure the placement performed by our emitter.
        subprocess.run(["cc", "-O0", "-shared", "-fPIC", source, "-o", library], check=True)
        lib = ctypes.CDLL(str(library))
        result = np.zeros((4, 5))
        pointer = ctypes.POINTER(ctypes.c_double)
        lib.metric(
            result.ctypes.data_as(pointer),
            geometry.ctypes.data_as(pointer),
            None,
            None,
            None,
            None,
            None,
        )
        expected = np.zeros_like(result)
        for point in range(nq):
            transform = geometry[point] if curved else geometry
            expected += (
                weights[point] * (transform @ left[:, point]).T @ (transform @ right[:, point])
            )
        np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)
        counts.append(ctypes.c_int.in_dll(lib, "reads").value)
    assert counts[0] > 0
    if curved:
        assert counts[0] * 7 == counts[1] * 4
    else:
        assert counts[0] == counts[1]


@pytest.mark.skipif(shutil.which("cc") is None, reason="no C compiler")
@pytest.mark.parametrize(
    "kind",
    [
        "independent",
        "scalar_updates",
        "different_extent",
        "dependent",
        "indirect_dependency",
        "calls",
    ],
)
def test_reduction_fusion_preserves_dependencies_and_summation_order(tmp_path, kind):
    i, j = index("i"), index("j")
    a = runtime_array("a", (6,))
    b = runtime_array("b", (5 if kind == "different_extent" else 6,))
    first = index_sum(a[i], (i,))
    factor = {"dependent": first, "indirect_dependency": first * a[0]}.get(kind, num(1))
    second = index_sum(factor * b[j], (j,))
    # Force an intervening use of the first result before the second reduction.
    expression = (first * a[0]) * second
    if kind == "scalar_updates":
        k = index("component")
        expression = index_sum(tensor((first, second))[k], (k,)) * a[0]

    def read_value(label, offset):
        access = f"w[{0 if label == 'a' else 6} + ({offset})]"
        return f"identity({access})" if kind == "calls" else access

    separate = c_kernel(expression, "separate", read_value, fuse_reductions=False)
    fused = c_kernel(expression, "fused", read_value)
    assert separate.count("for (") == 2
    assert fused.count("for (") == (1 if kind in ("independent", "scalar_updates") else 2)
    source, library = tmp_path / "fusion.c", tmp_path / "fusion.so"
    source.write_text("double identity(double x) { return x; }\n" + separate + fused)
    subprocess.run(
        ["cc", "-std=c99", "-O0", "-ffp-contract=off", "-shared", "-fPIC", source, "-o", library],
        check=True,
    )
    lib = ctypes.CDLL(str(library))
    pointer = ctypes.POINTER(ctypes.c_double)
    # Cancellation makes a changed summation order observable, even without fast math.
    samples = [np.array([1.0, 1e16, -1e16, 3.0, -2.0, 4.0] * 2)]
    samples.extend(np.random.default_rng(17).normal(size=(8, 12)))
    for sample, data in enumerate(samples):
        outputs = []
        for name in ("separate", "fused"):
            result = np.array([7.0])
            getattr(lib, name)(
                result.ctypes.data_as(pointer),
                data.ctypes.data_as(pointer),
                None,
                None,
                None,
                None,
                None,
            )
            outputs.append(result)
        np.testing.assert_array_equal(outputs[0], outputs[1])
        # The NumPy evaluator may reassociate the cancellation-sensitive sample.
        if sample:
            np.testing.assert_allclose(
                outputs[1],
                7 + evaluate(expression, {"a": data[:6], "b": data[6 : 6 + b.value_shape[0]]}),
                rtol=1e-12,
                atol=1e-12,
            )


@pytest.mark.skipif(shutil.which("cc") is None, reason="no C compiler")
@pytest.mark.parametrize("buffer_rows", [False, True])
@pytest.mark.parametrize("quadrature", [False, True])
def test_dense_row_buffers_preserve_rectangular_layout_and_accumulation(
    tmp_path, buffer_rows, quadrature
):
    from uflx_compiler.quadrature import constant

    i, r, j, s, q = (index(name) for name in ("i", "r", "j", "s", "q"))
    rng = np.random.default_rng(73)
    basis = rng.normal(size=(4, 5))
    components = rng.normal(size=(2, 3))
    coefficient = runtime_array("w", (2,) if quadrature else ())
    expression = constant(basis)[i, j] * constant(components)[r, s]
    expression = (
        index_sum(expression * coefficient[q], (q,)) if quadrature else expression * coefficient
    )
    code = c_kernel(
        expression,
        "blocked",
        lambda label, offset: f"w[{offset}]",
        output_indices=(i, r, j, s),
        unroll_indices=(r, s),
        buffer_rows=buffer_rows,
    )
    assert ("double rows" in code) == (buffer_rows and not quadrature)
    source, library = tmp_path / "buffered.c", tmp_path / "buffered.so"
    source.write_text(code)
    subprocess.run(["cc", "-O0", "-shared", "-fPIC", source, "-o", library], check=True)
    data = np.array([1.25, -0.75])
    initial = rng.normal(size=(8, 15))
    result = initial.copy()
    pointer = ctypes.POINTER(ctypes.c_double)
    ctypes.CDLL(str(library)).blocked(
        result.ctypes.data_as(pointer), data.ctypes.data_as(pointer), None, None, None, None, None
    )
    expected = initial + (data.sum() if quadrature else data[0]) * np.kron(basis, components)
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)
