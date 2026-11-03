"""Regression examples in the order of the revised language chapters."""

from __future__ import annotations

from collections import namedtuple

import numpy as np
import pytest

from uflx_oracle import evaluate
from uflx_spec_interpreter import (
    RD,
    Complex,
    Float,
    Integer,
    ScalarType,
    Zero,
    action,
    cell_map,
    component,
    component_tensor,
    conjugate,
    conjugate_space,
    coordinate,
    determinant,
    directional_derivative,
    dual_space,
    form,
    function,
    gathered_index,
    gradient,
    image,
    index,
    index_sum,
    indexed,
    integral,
    lower,
    replace,
    tensor,
)
from uflx_spec_interpreter._src.nodes import Base, Determinant, Function, Index, Indexed
from uflx_spec_interpreter._src.traversal import _expressions
from uflx_toy.cells import reference_simplex
from uflx_toy.indices import index_table
from uflx_toy.runtime import runtime_array, runtime_extent
from uflx_toy.spaces import space
from uflx_toy.topology import static_topology


class Payload(Base):
    identifier = "test.Payload"
    payload: object


@pytest.mark.parametrize("payload", [None, True, [], {}, 1j, namedtuple("Pair", "a")(1)])
def test_construction_arguments_are_curated(payload):
    with pytest.raises(TypeError):
        Payload((payload,))


@pytest.mark.parametrize("payload", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_arguments_are_rejected_even_in_labels(payload):
    with pytest.raises(ValueError):
        Index((payload,))


def test_constructor_arguments_are_positional_and_typed():
    with pytest.raises(TypeError):
        Integer(value=1)
    assert Float(-0.0) != Float(0.0)
    assert Payload(1) != Payload(1.0)
    with pytest.raises(TypeError):
        tensor([Integer(1)])


def test_complex_is_a_terminal_and_stays_complex():
    z = Complex(Integer(2), Float(-3.0))
    assert _expressions(z) == []
    assert z.scalar_type is ScalarType.complex
    assert evaluate(z, {})(()) == 2 - 3j
    assert lower(Complex(Integer(1), Integer(0))).scalar_type is ScalarType.complex


def test_empty_index_factories_and_nonempty_nodes():
    f = Integer(2)
    assert indexed(f, ()) is component_tensor(f, ()) is index_sum(f, ()) is f
    A = tensor((f, f))
    assert component(A, ()) is A
    with pytest.raises(IndexError):
        indexed(A, ())
    with pytest.raises(ValueError):
        component_tensor(A, ())
    with pytest.raises(ValueError):
        Indexed(f, ())
    with pytest.raises(TypeError):
        indexed(A, [0])


def test_gathered_indices_follow_direct_indices_in_static_order():
    i, j = index("i"), index("j")
    g = gathered_index(index_table((0, 1), 2), (i,))
    A = runtime_array("A", (2, 3))
    assert indexed(A, (g, j)).free_indices == ((j, 3), (i, 2))
    with pytest.raises(ValueError):
        indexed(A, (j, j))
    with pytest.raises(IndexError):
        indexed(A, (2, 0))


def test_determinants_are_nodes_including_symbolic_and_empty_shapes():
    n = runtime_extent("n")
    assert isinstance(determinant(runtime_array("A", (n, n))), Determinant)
    d = determinant(Zero((0, 0), (), (), ScalarType.real))
    assert lower(d) is d
    assert evaluate(d, {})(()) == 1
    with pytest.raises(ValueError):
        determinant(runtime_array("rectangular", (2, 3)))


def test_calculus_domain_is_the_result_domain():
    P = RD(2)
    X = coordinate(P)
    e = integral(X, P)
    assert e.domain is None and e.integration_domain is P
    assert gradient(Integer(1), P).domain is P
    with pytest.raises(ValueError):
        gradient(X, RD(3))
    with pytest.raises(ValueError):
        X[0] + coordinate(RD(3))[0]


def test_mesh_cells_retain_selection_after_geometry_substitution():
    T = static_topology((reference_simplex(1),), (((0, 1), (2, 3)),), ())
    V = space("coordinates", (1,), T, ScalarType.real)
    x, y = function(V, "x"), function(V, "y")
    D, E = image(x, RD(1)), image(y, RD(1))
    C = cell_map(D, 0, 1)
    assert C.entry == 1 and C.free_indices == ()
    assert C.parameterisation.map is cell_map(T, 0, 1)
    assert replace(C, {x: y}) is cell_map(E, 0, 1)
    with pytest.raises(ValueError):
        cell_map(D, -1)
    with pytest.raises(ValueError):
        cell_map(D, 0, 2)


def test_space_factories_and_substitution_keep_canonical_forms():
    V = space("V")
    R = space("R", scalar_type=ScalarType.real)
    assert conjugate_space(conjugate_space(V)) is V
    assert dual_space(dual_space(V)) is V
    assert dual_space(dual_space((V, V))) == (V, V)
    assert replace(conjugate_space(V), {V: R}) is R
    u = function(V, "u")
    assert replace(u, {V: R}) is function(R, "u")


def test_action_conjugation_and_real_to_complex_embedding():
    V = space("V")
    u = function(V, "u")
    e = action(conjugate(form(u, (u,))), 0, Complex(Integer(2), Integer(3)))
    assert evaluate(lower(e), {})(()) == 2 + 3j
    f = action(form(u ** Float(0.5), (u,)), 0, Integer(-1))
    assert lower(f).scalar_type is ScalarType.complex
    assert np.isclose(evaluate(lower(f), {})(()), 1j)


def test_derivative_substitutes_the_point_but_leaves_the_direction_alone():
    V = space("R", scalar_type=ScalarType.real)
    u = function(V, "u")
    e = directional_derivative(u * u, u, Integer(3), u)
    assert np.isclose(evaluate(lower(e), {u: 2})(()), 12)


def test_replacement_is_simultaneous_and_dictionary_order_independent():
    V = space("R", scalar_type=ScalarType.real)
    u, v, w = (function(V, name) for name in ("u", "v", "w"))
    assert replace(u, {u: v, v: w}) is v
    assert replace(u + v, {u + v: w, u: Integer(1)}) is w
    e = form(u + w, (v,))
    a = replace(e, {u: v, w: Integer(1)})
    b = replace(e, {w: Integer(1), u: v})
    assert a is b
    assert isinstance(a.arguments[0], Function) and a.arguments[0] is not v


@pytest.mark.parametrize("symbolic", [False, True])
def test_empty_sum_does_not_read_the_summand(symbolic):
    from uflx_compiler import evaluate as compiled_value

    n = runtime_extent("empty") if symbolic else 0
    i = index("empty-index")
    f = runtime_array("unavailable", (n,))
    total = index_sum(Integer(2) * f[i], (i,))
    env = {"empty": 0, n: 0} if symbolic else {}
    assert evaluate(total, env)(()) == 0
    assert compiled_value(total, env) == 0


def test_substituting_into_a_zero_selects_its_family_indices():
    i, j = index("zero-i"), index("zero-j")
    z = Zero((), ((i, 2),), (), ScalarType.real)
    assert replace(z, {i: 0}).free_indices == ()
    assert replace(z, {i: j}).free_indices == ((j, 2),)
    g = gathered_index(index_table((1, 0, 1), 2), (j,))
    assert replace(z, {i: g}).free_indices == ((j, 3),)


def test_zero_dimensional_coordinate_gradient_has_empty_shape():
    e = gradient(coordinate(RD(0)), RD(0))
    out = lower(e)
    assert out.value_shape == (0, 0)
    assert evaluate(out, {})(()).shape == (0, 0)
