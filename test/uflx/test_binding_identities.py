# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT
"""Identity contracts for factories, substitution and converging lowering paths."""

import pytest

from uflx._src.algorithms.substitution import substitute, substitute_values
from uflx._src.geometry import _metric
from uflx._src.nodes import Function, Index
from uflx._src.passes.lowering import lower_spatial
from uflx.algorithms import replace
from uflx.arithmetic import add, multiply, num
from uflx.calculus import directional_derivative, gradient
from uflx.forms import form, function, partial
from uflx.geometry import domain_map, jacobian
from uflx.nodes import EuclideanSpace, ScalarType, Zero
from uflx.tensors import index, index_sum, matrix_product, transpose

from .spaces import space

P = EuclideanSpace(2)
V = space("identity-matrix", P, value_shape=(2, 2))
A, B, C = (function(V, label) for label in ("A", "B", "C"))
W = space("identity-coordinate", P, value_shape=(2,), scalar_type=ScalarType.real)
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


@pytest.mark.parametrize("factory", [jacobian, _metric])
def test_geometry_identity_follows_parameterisation(factory):
    original = factory(D)
    assert factory(D) is original
    assert substitute(original, {x: y}) is factory(E)


@pytest.mark.parametrize(
    "factory, source, target",
    [
        (
            lambda a: gradient(a * a, P),
            function(space("scalar", P), "u"),
            function(space("scalar", P), "v"),
        ),
        (lambda a: gradient(transpose(a), P), A, B),
        (lambda a: jacobian(domain_map(P, a, P)), x, y),
        (lambda a: _metric(domain_map(P, a, P)), x, y),
    ],
    ids=["scalar-gradient", "tensor-gradient", "jacobian", "metric"],
)
def test_lowering_paths_converge_structurally(factory, source, target):
    original = factory(source)
    lowered = lower_spatial(original)
    expected = lower_spatial(factory(target))
    assert lower_spatial(original) is lowered
    assert lower_spatial(lowered) is lowered
    assert lower_spatial(substitute(original, {source: target})) is expected
    assert substitute(lowered, {source: target}) is expected


def test_independent_intermediate_construction_shares_nodes():
    intermediate = transpose(A)
    first = matrix_product(intermediate, A)
    second = matrix_product(substitute(transpose(B), {B: A}), A)
    assert first is second
    assert add(first, second).first is first


def test_capture_renaming_is_repeatable_and_then_fixed():
    V = space("identity-scalar", None)
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


def test_substitution_freshens_only_the_term_with_a_match():
    V = space("R", None, scalar_type=ScalarType.real)
    x, w, z = (function(V, name) for name in "xwz")
    F, H = form(x, (x,)), form(w * x, (x,))
    assert replace(F, {w: x}) is F
    result = replace(F + H, {w: x, z: x})
    assert result.first is F
    assert partial(result.first, {x: num(2)}) is num(2)
    assert result.second is replace(H, {w: x})
    (fresh,) = result.second.arguments
    assert fresh is not x
    assert result.second.expression is x * fresh


def test_substitution_uses_only_eligible_outermost_matches_in_scope():
    V = space("R", None, scalar_type=ScalarType.real)
    x, w, z = (function(V, name) for name in "xwz")
    body = w + z
    assert replace(form(x * body, (x,)), {body: num(3), w: x}) is form(x * num(3), (x,))
    nested = form(form(w, (w,)), (x,))
    assert replace(nested, {w: x}) is nested
    derivative = directional_derivative(x * x, x, w, num(1))
    assert replace(derivative, {w: x}) is directional_derivative(x * x, x, x, num(1))


def test_label_reconstruction_and_original_matches_under_a_binder():
    V = space("R", None, scalar_type=ScalarType.real)
    x, w = (function(V, name) for name in "xw")
    tagged_w = function(V, ("tag", w))
    tagged_x = function(V, ("tag", x))
    original = form(tagged_w, (x,))
    assert replace(original, {w: x}) is form(tagged_x, (x,))
    assert substitute_values(original, {w: x}) is original
    # Renaming x first would also rename x inside the key's label and lose the match.
    result = replace(form(x * tagged_x, (x,)), {tagged_x: x})
    (fresh,) = result.arguments
    assert fresh is not x
    assert result.expression is fresh * x


@pytest.mark.parametrize("substitution", [replace, substitute_values])
def test_index_substitution_freshens_for_a_used_zero_family_index(substitution):
    i, j, k = (index(name) for name in "ijk")
    summed = index_sum(Zero((), ((i, 2), (j, 2)), (), ScalarType.real), (i,))
    assert substitution(summed, {k: i}) is summed
    result = substitution(summed, {j: i})
    assert result.indices != (i,)
    assert result.free_indices == ((i, 2),)


def test_derived_index_dependencies_rewrite_without_changing_layout():
    i = index(("uflx.transpose", A, 0))
    assert substitute(i, {A: B}) is Index(("uflx.transpose", B, 0))
