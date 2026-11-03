"""Identity contracts for factories, substitution and converging lowering paths."""

import pytest

from uflx_spec_interpreter import (
    RD,
    ScalarType,
    add,
    domain_map,
    form,
    function,
    gradient,
    index,
    jacobian,
    matrix_product,
    multiply,
    num,
    transpose,
)
from uflx_spec_interpreter._src.geometry import metric
from uflx_spec_interpreter._src.nodes import Function, Index
from uflx_spec_interpreter._src.passes.gradients import lower
from uflx_spec_interpreter._src.substitution import substitute
from uflx_toy.spaces import space

P = RD(2)
V = space("identity-matrix", value_shape=(2, 2), domain=P)
A, B, C = (function(V, label) for label in ("A", "B", "C"))
W = space("identity-coordinate", value_shape=(2,), domain=P, scalar_type=ScalarType.real)
x, y = function(W, "x"), function(W, "y")
D, E = domain_map(P, x, P), domain_map(P, y, P)


@pytest.mark.parametrize(
    "factory",
    [
        transpose,
        lambda a: matrix_product(transpose(a), a),
        lambda a: add(a, a),
        lambda a: multiply(num(2), a),
    ],
    ids=["transpose", "matrix-product", "tensor-add", "scalar-multiply"],
)
def test_factory_and_substitution_share_the_same_node(factory):
    original = factory(A)
    expected = factory(B)
    assert factory(A) is original
    assert substitute(original, {A: B}) is expected
    assert substitute(substitute(original, {A: B}), {B: C}) is factory(C)


@pytest.mark.parametrize("factory", [jacobian, metric])
def test_geometry_identity_follows_parameterisation(factory):
    original = factory(D)
    assert factory(D) is original
    assert substitute(original, {x: y}) is factory(E)


@pytest.mark.parametrize(
    "factory, source, target",
    [
        (
            lambda a: gradient(a * a, P),
            function(space("scalar", domain=P), "u"),
            function(space("scalar", domain=P), "v"),
        ),
        (lambda a: gradient(transpose(a), P), A, B),
        (lambda a: jacobian(domain_map(P, a, P)), x, y),
        (lambda a: metric(domain_map(P, a, P)), x, y),
    ],
    ids=["scalar-gradient", "tensor-gradient", "jacobian", "metric"],
)
def test_lowering_paths_converge_structurally(factory, source, target):
    original = factory(source)
    lowered = lower(original)
    expected = lower(factory(target))
    assert lower(original) is lowered
    assert lower(lowered) is lowered
    assert lower(substitute(original, {source: target})) is expected
    assert substitute(lowered, {source: target}) is expected


def test_independent_intermediate_construction_shares_nodes():
    intermediate = transpose(A)
    first = matrix_product(intermediate, A)
    second = matrix_product(substitute(transpose(B), {B: A}), A)
    assert first is second
    assert add(first, second).expression.first.expression is first


def test_capture_renaming_is_repeatable_and_then_fixed():
    V = space("identity-scalar")
    u, v, w = (function(V, name) for name in ("u", "v", "w"))
    expression = form(u * v, (v,))
    renamed = substitute(expression, {u: v})
    assert substitute(expression, {u: v}) is renamed
    bound = renamed.arguments[0]
    assert bound is Function(
        V, ("uflx.rename", v, (expression.expression, expression.arguments), (v,))
    )
    changed = substitute(renamed, {v: w})
    assert changed.arguments[0] is bound
    # A fixed label still permits direct substitution of the named object itself.
    assert substitute(bound, {bound: w}) is w


def test_derived_index_dependencies_rewrite_without_changing_layout():
    i = index(("uflx.transpose", A, 0))
    assert substitute(i, {A: B}) is Index(("uflx.transpose", B, 0))
