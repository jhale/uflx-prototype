"""Metadata and DAG-sharing regressions for lexical alpha comparison."""

import sys

import pytest

from uflx_spec_interpreter import RD, form, function, image, index, integral, num
from uflx_spec_interpreter._src.algorithms.binding import alpha_equivalent
from uflx_spec_interpreter._src.algorithms.substitution import rename_bound
from uflx_spec_interpreter._src.extensions.binding import rebind
from uflx_toy.cells import reference_simplex
from uflx_toy.spaces import space
from uflx_toy.topology import static_topology


def geometry():
    K = reference_simplex(1)
    topology = static_topology((K,), (((0, 1),),), ())
    X = space("alpha-coordinates", value_shape=(1,), domain=topology, scalar_type="real")
    x, y = function(X, "x"), function(X, "y")
    V = space("alpha-physical", domain=image(x, RD(1)), scalar_type="real")
    return x, y, function(V, "c")


def test_geometry_dependent_coefficient_after_public_rebind():
    x, y, c = geometry()
    left = form(c, (x,))
    right = rebind(left, {x: y})
    assert right.expression.space.domain.coordinate_function is y
    assert alpha_equivalent(left, right)
    assert alpha_equivalent(right, left)
    assert rebind(right, {y: x}) is left
    # Without the outer binding correspondence these are different fields.
    assert not alpha_equivalent(left.expression, right.expression)


@pytest.mark.parametrize("used", [False, True])
def test_inner_parameter_space_follows_outer_binding(used):
    x, y, c = geometry()
    left = form(form(c if used else num(1), (c,)), (x,))
    right = rename_bound(left, {x: y})
    assert alpha_equivalent(left, right)
    assert alpha_equivalent(right, left)
    # Comparison does not relax the public application's dependent-space rules.
    with pytest.raises(ValueError, match="dependent parameter spaces"):
        rebind(left, {x: y})


def test_free_identity_and_space_differences_are_preserved():
    x, _, c = geometry()
    left = form(c, (x,))
    other_label = function(c.space, "another-coefficient")
    other_space = function(space("another-space", domain=c.domain, scalar_type="real"), "c")
    assert not alpha_equivalent(left, form(other_label, (x,)))
    assert not alpha_equivalent(left, form(other_space, (x,)))
    # A bound name does not make arbitrary differences in parameter types equal.
    assert not alpha_equivalent(
        form(form(c, (c,)), (x,)), form(form(other_space, (other_space,)), (x,))
    )


def test_generated_labels_keep_existing_identity_policy():
    V = space("alpha-labels", scalar_type="real")
    u, v = function(V, "u"), function(V, "v")
    c = function(V, ("library.coefficient", u))
    left = form(c, (u,))
    right = rebind(left, {u: v})
    assert right.expression.label == ("library.coefficient", v)
    assert not alpha_equivalent(left, right)
    assert alpha_equivalent(left, left)


@pytest.mark.parametrize("kind", ["function", "index"])
@pytest.mark.parametrize(
    ("first", "second"),
    [
        (("tag", 1), ("tag", 1.0)),
        (("tag", 0.0), ("tag", -0.0)),
        (("tag", (1,)), ("tag", (1.0,))),
        (("uflx.rename", 1), ("uflx.rename", 1.0)),
    ],
)
def test_free_labels_preserve_exact_construction_identity(kind, first, second):
    V = space("alpha-label-identity", scalar_type="real")
    make = (lambda label: function(V, label)) if kind == "function" else index
    left, right = make(first), make(second)
    assert left != right
    assert not alpha_equivalent(left, right)
    assert not alpha_equivalent(right, left)
    assert alpha_equivalent(left, make(first))
    if kind == "function":
        # Once bound, the spelling of these labels must still be immaterial.
        assert alpha_equivalent(form(left, (left,)), form(right, (right,)))


def test_nodes_embedded_in_labels_remain_opaque():
    V = space("alpha-opaque-label", scalar_type="real")
    u, v = function(V, "u"), function(V, "v")
    f, g = form(u, (u,)), form(v, (v,))
    assert alpha_equivalent(f, g)
    assert not alpha_equivalent(function(V, ("label", f)), function(V, ("label", g)))


def test_deep_shared_graph_without_changing_recursion_limit():
    V = space("alpha-deep", scalar_type="real")
    u, v, w = (function(V, name) for name in ("u", "v", "w"))
    left, right, wrong = u, v, w
    for _ in range(2000):
        left, right, wrong = left + left, right + right, wrong + wrong
    limit = sys.getrecursionlimit()
    assert alpha_equivalent(form(left, (u,)), form(right, (v,)))
    assert not alpha_equivalent(form(left, (u,)), form(wrong, (v,)))
    assert sys.getrecursionlimit() == limit


def test_shared_nodes_keep_relevant_outer_bindings():
    V = space("alpha-shadowing", scalar_type="real")
    u, v, w, p = (function(V, name) for name in ("u", "v", "w", "p"))
    left = form(form(u, (v,)) + form(u, (u,)), (u,))
    right = form(form(w, (p,)) + form(p, (p,)), (w,))
    wrong = form(form(p, (p,)) + form(p, (p,)), (w,))
    assert alpha_equivalent(left, right)
    assert not alpha_equivalent(left, wrong)
    assert not alpha_equivalent(right, wrong)


def test_parameter_metadata_is_relevant_to_cache_context():
    x, y, c = geometry()
    # The same c is encountered once with x bound and once with x free.
    body = integral(c, c.domain)
    left = form(body, (x,)) + form(body, (y,))
    right_first = rebind(form(body, (x,)), {x: y})
    right = right_first + form(body, (x,))
    assert not alpha_equivalent(left, right)


def test_irrelevant_scopes_do_not_expand_shared_dag():
    V = space("alpha-dag", scalar_type="real")
    u, v, w, p = (function(V, name) for name in ("u", "v", "w", "p"))

    def visits(depth):
        left, right = num(1), num(1)
        for _ in range(depth):
            left = form(left, (u,)) + form(left, (v,))
            right = form(right, (w,)) + form(right, (p,))
        count = 0

        def profile(frame, event, arg):
            nonlocal count
            if (
                event == "call"
                and frame.f_code.co_filename == alpha_equivalent.__code__.co_filename
                and frame.f_code.co_name in ("compare", "compare_nodes", "node_dependencies")
            ):
                count += 1

        previous = sys.getprofile()
        try:
            sys.setprofile(profile)
            assert alpha_equivalent(left, right)
        finally:
            sys.setprofile(previous)
        return count

    small, large = visits(6), visits(12)
    # Count work rather than wall-clock time. Whole-scope memoisation grows
    # approximately 64-fold here; projected contexts should grow linearly.
    assert small > 0
    assert large <= 3 * small
