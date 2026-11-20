import ctypes
import re
import shutil
import subprocess

import basix
import numpy as np
import pytest
from testutil import grouped_topology, push_forward

from uflx_compiler import compile_form, evaluate
from uflx_compiler.c_code import c_kernel
from uflx_spec_interpreter import (
    EuclideanSpace,
    broadcast_to,
    component_tensor,
    divergence,
    function,
    gathered_index,
    gradient,
    index,
    index_sum,
    integral,
    inverse,
    logarithm,
    mesh,
    num,
    square_root,
    tensor,
)
from uflx_toy.basix_cells import basix_cell
from uflx_toy.basix_spaces import basix_space
from uflx_toy.indices import index_table
from uflx_toy.runtime import runtime_array

P, RT = basix.ElementFamily.P, basix.ElementFamily.RT
T = grouped_topology((basix_cell(basix.CellType.triangle),), (((0, 1, 2),),), ())
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
    D = mesh(tensor((function(X, "x0"), function(X, "x1"))), EuclideanSpace(2))
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
    assert loops == ["q", "i"]  # Small basis loops are specialised inside quadrature.
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
@pytest.mark.parametrize("specialize", [False, True])
def test_component_specialisation_skips_zero_blocks(tmp_path, quadrature, specialize):
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
        specialize_basis=specialize,
    )
    # Two diagonal blocks are emitted; the two identically-zero blocks do no work.
    specialized = specialize
    assert code.count("A[") == (18 if specialized else 2)
    # Both component updates share basis loops, unless their entries are specialised.
    assert len(re.findall(r"for \(int a\d+ =", code)) == (0 if specialized else 2)
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
def test_reductions_preserve_dependencies_and_summation_order(tmp_path, kind):
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

    code = c_kernel(expression, "reductions", read_value)
    source, library = tmp_path / "reductions.c", tmp_path / "reductions.so"
    source.write_text("double identity(double x) { return x; }\n" + code)
    subprocess.run(
        ["cc", "-std=c99", "-O0", "-ffp-contract=off", "-shared", "-fPIC", source, "-o", library],
        check=True,
    )
    lib = ctypes.CDLL(str(library))
    pointer = ctypes.POINTER(ctypes.c_double)
    # Cancellation makes a changed summation order observable, even without fast math.
    samples = [np.array([1.0, 1e16, -1e16, 3.0, -2.0, 4.0] * 2)]
    samples.extend(np.random.default_rng(17).normal(size=(8, 12)))
    for data in samples:
        result = np.array([7.0])
        lib.reductions(
            result.ctypes.data_as(pointer),
            data.ctypes.data_as(pointer),
            None,
            None,
            None,
            None,
            None,
        )
        first = 0.0
        for value in data[:6]:
            first += float(value)
        factor = {"dependent": first, "indirect_dependency": first * data[0]}.get(kind, 1.0)
        second = 0.0
        for value in data[6 : 6 + b.value_shape[0]]:
            second += factor * float(value)
        expected = (
            (first + second) * data[0] if kind == "scalar_updates" else (first * data[0]) * second
        )
        np.testing.assert_array_equal(result, [7 + expected])


@pytest.mark.skipif(shutil.which("cc") is None, reason="no C compiler")
@pytest.mark.parametrize("quadrature", [False, True])
def test_dense_blocks_preserve_rectangular_layout_and_accumulation(tmp_path, quadrature):
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
    )
    source, library = tmp_path / "blocked.c", tmp_path / "blocked.so"
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


@pytest.mark.skipif(shutil.which("cc") is None, reason="no C compiler")
def test_coupled_blocks_prepare_geometry_products_before_basis_loops(tmp_path):
    from uflx_compiler.quadrature import constant

    counts = []
    for nj in (4, 7):
        i, r, j, s, a, b = (index(name) for name in ("i", "r", "j", "s", "a", "b"))
        rng = np.random.default_rng(83)
        left, right = rng.normal(size=(2, 3)), rng.normal(size=(2, nj))
        geometry = rng.normal(size=(2, 2))
        G = runtime_array("geometry", geometry.shape)
        expression = index_sum(G[r, a] * constant(left)[a, i], (a,)) * index_sum(
            G[s, b] * constant(right)[b, j], (b,)
        )
        code = c_kernel(
            expression,
            "coupled",
            lambda label, offset: f"geometry_read(w, {offset})",
            output_indices=(i, r, j, s),
            unroll_indices=(r, s),
        )
        source, library = tmp_path / f"coupled{nj}.c", tmp_path / f"coupled{nj}.so"
        source.write_text(
            "int reads = 0;\n"
            "double geometry_read(const double* w, int k) { reads++; return w[k]; }\n" + code
        )
        # Measure the scheduling done by our emitter, without C optimisation.
        subprocess.run(["cc", "-O0", "-shared", "-fPIC", source, "-o", library], check=True)
        lib = ctypes.CDLL(str(library))
        result = np.full((6, 2 * nj), 7.0)
        pointer = ctypes.POINTER(ctypes.c_double)
        lib.coupled(
            result.ctypes.data_as(pointer),
            geometry.ctypes.data_as(pointer),
            None,
            None,
            None,
            None,
            None,
        )
        expected = np.einsum("ri,sj->irjs", geometry @ left, geometry @ right).reshape(6, 2 * nj)
        np.testing.assert_allclose(result, 7 + expected, rtol=1e-12, atol=1e-12)
        counts.append(ctypes.c_int.in_dll(lib, "reads").value)
    assert counts[0] > 0
    assert counts[0] == counts[1]


@pytest.mark.skipif(shutil.which("cc") is None, reason="no C compiler")
@pytest.mark.parametrize("specialize", [False, True])
@pytest.mark.parametrize("columns", [3, 4])
@pytest.mark.parametrize("quadrature", [False, True])
def test_small_basis_specialisation_eliminates_zero_updates(
    tmp_path, specialize, columns, quadrature
):
    from uflx_compiler.quadrature import constant

    i, j = index("i"), index("j")
    left, right = np.array([0.0, 1.0, -1.0]), np.arange(columns, dtype=float)
    scale = runtime_array("scale", ())
    expression = scale * constant(left)[i] * constant(right)[j]
    data = np.array([2.5])
    if quadrature:
        q = index("q")
        data = np.array([2.5, -0.75, 1.25, 0.5])
        expression = index_sum(
            runtime_array("weights", (4,))[q] * constant(left)[i] * constant(right)[j], (q,)
        )
    code = c_kernel(
        expression,
        "small",
        lambda label, offset: f"w[{offset}]",
        output_indices=(i, j),
        specialize_basis=specialize,
    )
    specialized = specialize and columns == 3
    assert ("for (int a" not in code) == specialized
    if specialized:
        assert code.count("A[") == 4  # Only the nonzero rows and columns are written.
    source, library = tmp_path / "small.c", tmp_path / "small.so"
    source.write_text(code)
    subprocess.run(
        ["cc", "-O0", "-ffp-contract=off", "-shared", "-fPIC", source, "-o", library],
        check=True,
    )
    result = np.full((3, columns), 7.0)
    pointer = ctypes.POINTER(ctypes.c_double)
    ctypes.CDLL(str(library)).small(
        result.ctypes.data_as(pointer), data.ctypes.data_as(pointer), None, None, None, None, None
    )
    np.testing.assert_array_equal(result, 7 + data.sum() * np.outer(left, right))


def test_small_basis_specialisation_respects_total_update_budget():
    from uflx_compiler.quadrature import constant

    i, r, j, s = (index(name) for name in ("i", "r", "j", "s"))
    basis = constant(np.array([0.0, 1.0, -1.0]))
    components = constant(np.ones((3, 3)))
    expression = runtime_array("scale", ()) * basis[i] * basis[j] * components[r, s]
    code = c_kernel(
        expression,
        "large_blocks",
        lambda label, offset: f"w[{offset}]",
        output_indices=(i, r, j, s),
        unroll_indices=(r, s),
    )
    # Nine blocks of nine entries exceed the 64-update specialisation budget.
    assert "for (int a" in code


@pytest.mark.skipif(shutil.which("cc") is None, reason="no C compiler")
@pytest.mark.parametrize("reuse", [False, True])
@pytest.mark.parametrize("kind", ["symmetric", "different_material", "different_basis"])
def test_transpose_reuse_requires_equal_blocks_and_preserves_existing_output(tmp_path, reuse, kind):
    from uflx_compiler.quadrature import constant

    q, i, r, j, s = (index(name) for name in ("q", "i", "r", "j", "s"))
    rng = np.random.default_rng(81)
    basis = rng.normal(size=(4, 5))
    right = rng.normal(size=(4, 5)) if kind == "different_basis" else basis
    blocks = np.array([[2.0, 3.0], [4.0 if kind == "different_material" else 3.0, 5.0]])
    weights = runtime_array("weights", (4,))
    expression = index_sum(
        weights[q] * constant(basis)[q, i] * constant(right)[q, j] * constant(blocks)[r, s],
        (q,),
    )
    code = c_kernel(
        expression,
        "symmetric",
        lambda label, offset: f"w[{offset}]",
        output_indices=(i, r, j, s),
        unroll_indices=(r, s),
        reuse_transposes=reuse,
    )
    assert ("double local" in code) == (reuse and kind == "symmetric")
    source, library = tmp_path / "symmetric.c", tmp_path / "symmetric.so"
    source.write_text(code)
    subprocess.run(["cc", "-O0", "-shared", "-fPIC", source, "-o", library], check=True)
    data = rng.normal(size=4)
    initial = rng.normal(size=(10, 10))  # Intentionally not symmetric.
    actual = initial.copy()
    pointer = ctypes.POINTER(ctypes.c_double)
    ctypes.CDLL(str(library)).symmetric(
        actual.ctypes.data_as(pointer), data.ctypes.data_as(pointer), None, None, None, None, None
    )
    expected = initial + np.kron(basis.T @ (data[:, None] * right), blocks)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.skipif(shutil.which("cc") is None, reason="no C compiler")
def test_action_factors_geometry_before_test_basis_loop(tmp_path):
    from uflx_compiler.quadrature import constant

    counts = []
    for dofs in (4, 7):
        q, i, a = (index(name) for name in ("q", "i", "a"))
        rng = np.random.default_rng(46)
        basis = rng.normal(size=(2, 3, dofs))
        geometry = rng.normal(size=(3, 2, 2))
        G = runtime_array("G", geometry.shape)
        derivative = index_sum(G[q, 0, a] * constant(basis)[a, q, i], (a,))
        expression = index_sum(G[q, 1, 0] * derivative, (q,))
        code = c_kernel(
            expression,
            "action",
            lambda label, off: f"read_geometry(w, {off})",
            output_indices=(i,),
            preintegrate=False,
        )
        source, library = tmp_path / f"action{dofs}.c", tmp_path / f"action{dofs}.so"
        source.write_text(
            "int reads = 0;\n"
            "double read_geometry(const double* w, int k) { reads++; return w[k]; }\n" + code
        )
        subprocess.run(["cc", "-O0", "-shared", "-fPIC", source, "-o", library], check=True)
        lib = ctypes.CDLL(str(library))
        actual = np.full(dofs, 7.0)
        pointer = ctypes.POINTER(ctypes.c_double)
        lib.action(
            actual.ctypes.data_as(pointer),
            geometry.ctypes.data_as(pointer),
            None,
            None,
            None,
            None,
            None,
        )
        expected = 7 + np.einsum("q,qa,aqi->i", geometry[:, 1, 0], geometry[:, 0, :], basis)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
        counts.append(ctypes.c_int.in_dll(lib, "reads").value)
    assert counts[0] > 0
    assert counts[0] == counts[1]


@pytest.mark.parametrize("perturbation", [0.0, 2e-15, 1e-8])
def test_sparse_table_contractions_preserve_values_and_distinct_columns(perturbation):
    from uflx_compiler.contractions import _sparse_contractions
    from uflx_compiler.quadrature import constant
    from uflx_spec_interpreter._src.nodes import IndexSum

    q, k = index("q"), index("k")
    column = np.array([1.0, 2.0, -3.0])
    table = np.column_stack((column, -column * (1 + perturbation)))
    weights = runtime_array("weights", (2,))
    expression = index_sum(constant(table)[q, k] * weights[k], (k,))
    result = _sparse_contractions(expression)
    assert isinstance(result, IndexSum) == (perturbation == 1e-8)
    np.testing.assert_allclose(
        evaluate(result, {"weights": np.array([2.0, 3.0])}),
        table @ np.array([2.0, 3.0]),
        rtol=1e-13,
        atol=0,
    )
    assert result.free_indices == expression.free_indices


@pytest.mark.skipif(shutil.which("cc") is None, reason="no C compiler")
@pytest.mark.parametrize("enabled", [False, True])
def test_sparse_table_contractions_hoist_coefficients_and_skip_zero_columns(tmp_path, enabled):
    from uflx_compiler.quadrature import constant

    q, k, i = index("q"), index("k"), index("i")
    column = np.array([1.0, 2.0, -3.0, 4.0, -2.0])
    table = np.column_stack((column, column, -column, np.zeros(5)))
    expression = index_sum(constant(table)[q, k] * runtime_array("weights", (4,))[k], (k,))
    basis = np.arange(25.0).reshape(5, 5) + 1
    expression = index_sum(expression * expression * constant(basis)[q, i], (q,))
    code = c_kernel(
        expression,
        "sparse",
        lambda label, offset: f"read_weight(w[{offset}])",
        output_indices=(i,),
        preintegrate=False,
        sparse_tables=enabled,
    )
    source, library = tmp_path / "sparse.c", tmp_path / "sparse.so"
    source.write_text(
        "int reads = 0;\ndouble read_weight(double x) { reads++; return x; }\n" + code
    )
    subprocess.run(["cc", "-O0", "-shared", "-fPIC", source, "-o", library], check=True)
    lib = ctypes.CDLL(str(library))
    data = np.array([2.0, -0.75, 0.5, 99.0])
    actual = np.arange(5.0)
    pointer = ctypes.POINTER(ctypes.c_double)
    lib.sparse(
        actual.ctypes.data_as(pointer), data.ctypes.data_as(pointer), None, None, None, None, None
    )
    np.testing.assert_allclose(
        actual, np.arange(5.0) + basis.T @ (table @ data) ** 2, rtol=1e-13, atol=1e-13
    )
    if enabled:
        assert ctypes.c_int.in_dll(lib, "reads").value == 3


def test_sparse_table_contractions_preserve_tiny_and_all_zero_tables():
    from uflx_compiler.contractions import _sparse_contractions
    from uflx_compiler.quadrature import constant

    q, k = index("q"), index("k")
    for table in (np.zeros((3, 2)), np.array([[1e-100, 0], [2e-100, 0], [-3e-100, 0]])):
        expression = index_sum(constant(table)[q, k] * runtime_array("weights", (2,))[k], (k,))
        result = _sparse_contractions(expression)
        assert result.free_indices == expression.free_indices
        np.testing.assert_allclose(
            evaluate(result, {"weights": np.array([2.0, 3.0])}),
            table @ np.array([2.0, 3.0]),
            rtol=1e-14,
            atol=0,
        )


@pytest.mark.skipif(shutil.which("cc") is None, reason="no C compiler")
@pytest.mark.parametrize("kind", ["same_basis", "different_basis", "varying_basis"])
def test_small_constant_bases_collect_varying_geometry_into_metric(tmp_path, kind):
    from uflx_compiler.quadrature import constant

    q, a, i, j = (index(name) for name in ("q", "a", "i", "j"))
    rng = np.random.default_rng(217)
    geometry = rng.normal(size=(5, 2, 2))
    left = np.array([[-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]])
    right = left if kind == "same_basis" else np.array([[2.0, 0.0, -1.0], [-1.0, 1.0, 0.0]])
    if kind == "varying_basis":
        right = rng.normal(size=(5, 2, 3))
    G = runtime_array("geometry", geometry.shape)
    R = constant(right)[q, a, j] if kind == "varying_basis" else constant(right)[a, j]
    products = [
        index_sum(G[q, r, a] * constant(left)[a, i], (a,)) * index_sum(G[q, r, a] * R, (a,))
        for r in range(2)
    ]
    expression = index_sum(products[0] + products[1], (q,))
    code = c_kernel(
        expression,
        "metric",
        lambda label, offset: f"w[{offset}]",
        output_indices=(i, j),
        preintegrate=False,
        sparse_tables=False,
    )
    if kind != "varying_basis":
        assert "double sum" not in code  # No per-basis physical-gradient reductions.
    source, library = tmp_path / "metric.c", tmp_path / "metric.so"
    source.write_text(code)
    subprocess.run(["cc", "-O0", "-shared", "-fPIC", source, "-o", library], check=True)
    initial = rng.normal(size=(3, 3))
    actual = initial.copy()
    pointer = ctypes.POINTER(ctypes.c_double)
    ctypes.CDLL(str(library)).metric(
        actual.ctypes.data_as(pointer),
        geometry.ctypes.data_as(pointer),
        None,
        None,
        None,
        None,
        None,
    )
    L = np.einsum("qra,ai->qri", geometry, left)
    R = (
        np.einsum("qra,qaj->qrj", geometry, right)
        if kind == "varying_basis"
        else np.einsum("qra,aj->qrj", geometry, right)
    )
    np.testing.assert_allclose(
        actual, initial + np.einsum("qri,qrj->ij", L, R), rtol=1e-12, atol=1e-12
    )


@pytest.mark.skipif(shutil.which("cc") is None, reason="no C compiler")
@pytest.mark.parametrize("matrix", [False, True])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("nq", [1, 5])
def test_split_quadrature_preserves_reductions_and_accumulation(tmp_path, matrix, nested, nq):
    from uflx_compiler.quadrature import constant

    q, r, k, i, j = (index(name) for name in ("q", "r", "k", "i", "j"))
    rng = np.random.default_rng(721)
    coefficients = rng.normal(size=7)
    values = rng.normal(size=(nq, 7))
    left = rng.normal(size=(nq, 4))
    right = rng.normal(size=(nq, 6))
    weights = rng.normal(size=(3, nq))
    c = runtime_array("coefficient", (7,))
    factor = index_sum(c[k] * constant(values)[q, k], (k,))
    integrand = factor * constant(left)[q, i]
    if matrix:
        integrand = integrand * constant(right)[q, j]
    if nested:
        integrand = integrand * constant(weights)[r, q]
    expression = index_sum(integrand, (r, q) if nested else (q,))
    shape = (4, 6) if matrix else (4,)
    initial = rng.normal(size=shape)
    results = []
    for split in (False, True):
        code = c_kernel(
            expression,
            "split",
            lambda label, offset: f"w[{offset}]",
            output_indices=(i, j) if matrix else (i,),
            preintegrate=False,
            sparse_tables=False,
            split_quadrature=split,
        )
        assert ("double factor_" in code) == (split and nq > 1)
        source, library = tmp_path / f"split_{split}.c", tmp_path / f"split_{split}.so"
        source.write_text(code)
        subprocess.run(
            ["cc", "-O0", "-ffp-contract=off", "-shared", "-fPIC", source, "-o", library],
            check=True,
        )
        actual = initial.copy()
        pointer = ctypes.POINTER(ctypes.c_double)
        ctypes.CDLL(str(library)).split(
            actual.ctypes.data_as(pointer),
            coefficients.ctypes.data_as(pointer),
            None,
            None,
            None,
            None,
            None,
        )
        results.append(actual)
    factor = values @ coefficients
    if nested:
        factor *= weights.sum(axis=0)
    expected = (
        np.einsum("q,qi,qj->ij", factor, left, right)
        if matrix
        else np.einsum("q,qi->i", factor, left)
    )
    np.testing.assert_array_equal(*results)
    np.testing.assert_allclose(results[1], initial + expected, rtol=1e-12, atol=1e-12)


@pytest.mark.skipif(shutil.which("cc") is None, reason="no C compiler")
@pytest.mark.parametrize("dependent_denominator", [False, True])
def test_action_factoring_through_division(tmp_path, dependent_denominator):
    from uflx_compiler.quadrature import constant

    q, i = index("q"), index("i")
    rng = np.random.default_rng(931)
    matrix = rng.normal(size=(4, 3, 3))
    force = rng.normal(size=(4, 3))
    basis = rng.normal(size=(4, 3, 5))
    denominator = rng.uniform(0.5, 2, size=(4, 5) if dependent_denominator else (4,))
    G = runtime_array("matrix", matrix.shape)
    F = runtime_array("force", force.shape)
    D = runtime_array("denominator", denominator.shape)
    scale = D[q, i] if dependent_denominator else D[q]
    terms = []
    for r in range(3):
        components = sum((G[q, r, k] * constant(basis)[q, k, i] for k in range(3)), num(0))
        terms.append(F[q, r] * (components / scale))
    expression = index_sum(terms[0] + terms[1] + terms[2], (q,))
    offsets = {"matrix": 0, "force": matrix.size, "denominator": matrix.size + force.size}
    code = c_kernel(
        expression,
        "divided_action",
        lambda label, offset: f"w[{offsets[label]} + ({offset})]",
        output_indices=(i,),
        preintegrate=False,
        sparse_tables=False,
        split_quadrature=False,
    )
    # An independent divisor is applied before the test-basis loop. A divisor
    # depending on the test index must remain inside it.
    assert (" / " in code.split("for (int a0", 1)[1]) == dependent_denominator
    source, library = tmp_path / "action.c", tmp_path / "action.so"
    source.write_text(code)
    subprocess.run(["cc", "-O0", "-shared", "-fPIC", source, "-o", library], check=True)
    data = np.concatenate((matrix.ravel(), force.ravel(), denominator.ravel()))
    initial = rng.normal(size=5)
    actual = initial.copy()
    pointer = ctypes.POINTER(ctypes.c_double)
    ctypes.CDLL(str(library)).divided_action(
        actual.ctypes.data_as(pointer),
        data.ctypes.data_as(pointer),
        None,
        None,
        None,
        None,
        None,
    )
    values = np.einsum("qr,qrc,qci->qi", force, matrix, basis)
    expected = (values / (denominator if dependent_denominator else denominator[:, None])).sum(
        axis=0
    )
    np.testing.assert_allclose(actual, initial + expected, rtol=1e-12, atol=1e-12)


@pytest.mark.skipif(shutil.which("cc") is None, reason="no C compiler")
def test_division_factoring_keeps_reference_sums_factored(tmp_path):
    from uflx_compiler.contractions import _factor_contraction
    from uflx_compiler.quadrature import constant
    from uflx_spec_interpreter._src.algorithms.traversal import everywhere

    q, c, i, j = (index(name) for name in ("q", "c", "i", "j"))
    rng = np.random.default_rng(933)
    left, right = rng.normal(size=(5, 3, 4)), rng.normal(size=(5, 3, 6))
    denominator = rng.uniform(0.5, 2, size=5)
    D = runtime_array("denominator", (5,))
    L = index_sum(constant(left)[q, c, i], (c,))
    R = index_sum(constant(right)[q, c, j], (c,))
    expression = index_sum((L / D[q]) * (R / D[q]), (q,))
    code = c_kernel(
        expression,
        "reference_sums",
        lambda label, offset: f"w[{offset}]",
        output_indices=(i, j),
        preintegrate=False,
        sparse_tables=False,
    )
    factored = _factor_contraction(expression.expression, (i, j), (q,), combine_geometry=True)
    nodes = set()
    everywhere(factored, lambda n: not nodes.add(n))
    assert L in nodes and R in nodes  # Preserve sums instead of expanding nine products.
    source, library = tmp_path / "reference.c", tmp_path / "reference.so"
    source.write_text(code)
    subprocess.run(["cc", "-O0", "-shared", "-fPIC", source, "-o", library], check=True)
    initial = rng.normal(size=(4, 6))
    actual = initial.copy()
    pointer = ctypes.POINTER(ctypes.c_double)
    ctypes.CDLL(str(library)).reference_sums(
        actual.ctypes.data_as(pointer),
        denominator.ctypes.data_as(pointer),
        None,
        None,
        None,
        None,
        None,
    )
    expected = np.einsum("qi,qj,q->ij", left.sum(axis=1), right.sum(axis=1), 1 / denominator**2)
    np.testing.assert_allclose(actual, initial + expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("transpose", [False, True])
def test_division_factoring_preserves_unexpanded_contractions(transpose):
    from uflx_compiler.contractions import _factor_contraction
    from uflx_compiler.quadrature import constant
    from uflx_spec_interpreter._src.algorithms.traversal import everywhere

    q, c, i, j = (index(name) for name in ("q", "c", "i", "j"))
    rng = np.random.default_rng(935)
    G = runtime_array("material", (5, 2))
    D = runtime_array("denominator", (5,))
    row, col = (j, i) if transpose else (i, j)
    basis = constant(rng.normal(size=(5, 2, 4)))
    components = tensor(tuple(G[q, k] * basis[q, k, col] for k in range(2)))
    quotient = index_sum(components[c], (c,)) / D[q]
    expression = constant(rng.normal(size=(5, 6)))[q, row] * quotient
    factored = _factor_contraction(expression, (i, j), (q,), combine_geometry=True)
    nodes = set()
    everywhere(factored, lambda n: not nodes.add(n))
    # The matrix collector declined component expansion; division must not
    # bypass that scheduling decision by extracting a reciprocal anyway.
    assert quotient in nodes


@pytest.mark.skipif(shutil.which("cc") is None, reason="no C compiler")
@pytest.mark.parametrize("shape", [(4, 7), (3, 4, 5)])
@pytest.mark.parametrize("access", ["contiguous", "transposed", "row"])
def test_dense_output_flattening(tmp_path, shape, access):
    from uflx_compiler.quadrature import constant

    rng = np.random.default_rng(937)
    axes = tuple(index(f"i{k}") for k in range(len(shape)))
    values = rng.normal(size=shape)
    weight = runtime_array("weight", (1,))
    expression = weight[0] * constant(values)[axes]
    expected = 1.7 * values
    if access == "transposed":
        other = rng.normal(size=shape[::-1])
        expression = expression + constant(other)[axes[::-1]]
        expected = expected + other.transpose()
    elif access == "row":
        rows = runtime_array("rows", (shape[0],))
        expression = expression + rows[axes[0]]
        expected = expected + np.arange(shape[0]).reshape((-1,) + (1,) * (len(shape) - 1))
    code = c_kernel(
        expression,
        "dense",
        lambda label, offset: f"w[{offset}]" if label == "weight" else f"w[1 + {offset}]",
        output_indices=axes,
        specialize_basis=False,
        sparse_tables=False,
    )
    # Partial flattening of the inner axes is allowed when the row is used separately.
    assert (f"flat_a0 < {values.size}" in code) == (access == "contiguous")
    source, library = tmp_path / "dense.c", tmp_path / "dense.so"
    source.write_text(code)
    subprocess.run(["cc", "-O0", "-shared", "-fPIC", source, "-o", library], check=True)
    initial = rng.normal(size=shape)
    actual = initial.copy()
    data = np.r_[1.7, np.arange(shape[0], dtype=float)]
    pointer = ctypes.POINTER(ctypes.c_double)
    ctypes.CDLL(str(library)).dense(
        actual.ctypes.data_as(pointer),
        data.ctypes.data_as(pointer),
        None,
        None,
        None,
        None,
        None,
    )
    np.testing.assert_allclose(actual, initial + expected, rtol=1e-12, atol=1e-12)
