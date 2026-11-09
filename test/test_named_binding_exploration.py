"""Behavioral probes for the proposed binding interface, outside the public API."""

import pytest

from uflx_spec_interpreter import (
    RD,
    apply_actions,
    conjugate,
    directional_derivative,
    domain_map,
    dual_space,
    form,
    function,
    integral,
    num,
    pull_back,
)
from uflx_spec_interpreter._src.algorithms.substitution import (
    occurs,
    occurs_in_metadata,
    occurs_in_values,
    substitute,
)
from uflx_spec_interpreter._src.extensions.binding import alpha_equivalent, apply, partial, rebind
from uflx_spec_interpreter._src.nodes import Action
from uflx_toy.spaces import space

V = space("binding-probe", scalar_type="real")
u, v, w, p = (function(V, name) for name in ("u", "v", "w", "p"))


def test_partial_is_simultaneous_and_preserves_retained_names():
    f = form(u * v, (u, v))
    assert partial(f, {u: v, v: u}) is v * u
    assert partial(f, {u: w}) is form(w * v, (v,))
    assert partial(f, {u: v, v: u}) is partial(f, {v: u, u: v})
    assert partial(f, {}) is f


def test_retained_capture_requires_explicit_rebind():
    f = form(u * v, (u, v))
    with pytest.raises(ValueError, match="retained argument"):
        partial(f, {u: v})
    assert partial(rebind(f, {v: p}), {u: v}) is form(v * p, (p,))


def test_rebind_swaps_but_rejects_capture_and_duplicates():
    f = form(u * v, (u, v))
    renamed = rebind(f, {u: v, v: u})
    assert renamed is form(v * u, (v, u))
    assert alpha_equivalent(f, renamed)
    with pytest.raises(ValueError, match="duplicate"):
        rebind(f, {u: v})
    with pytest.raises(ValueError, match="capture"):
        rebind(form(u * v, (u,)), {u: v})


def test_nested_shadowing_and_internal_capture():
    inner = form(u * v, (v,))
    assert partial(form(inner, (v,)), {v: w}) is inner
    result = partial(form(inner, (u,)), {u: v})
    assert alpha_equivalent(result, form(v * p, (p,)))
    assert occurs(v, result)


def test_partial_rejects_ambiguous_maps_and_inner_selectors():
    f = form(u * v, (u, v))
    with pytest.raises(TypeError, match="explicit Form"):
        partial(f + f, {u: w})
    with pytest.raises(ValueError, match="outer argument"):
        partial(form(f, (w,)), {u: p})


def test_full_application_works_on_sums_and_nested_forms():
    f = form(u * v, (u, v))
    result = apply(f + f, (w, p))
    assert isinstance(result, Action)
    assert apply_actions(result) is w * p + w * p
    nested = form(form(u * v, (u,)), (v,))
    assert apply_actions(apply(nested, (w, p))) is w * p
    # The supplied v remains free while the second slot receives w.
    assert apply_actions(apply(f, (v, w))) is v * w
    with pytest.raises(ValueError, match="every argument"):
        apply(f, (w,))
    assert apply(num(1), ()) is num(1)


def test_mixed_entries_preserve_grouping():
    f = form(u + v, ((u, v),))
    assert partial(f, {(u, v): (w, p)}) is w + p
    assert rebind(f, {(u, v): (w, p)}) is form(w + p, ((w, p),))
    with pytest.raises(ValueError, match="outer argument"):
        partial(f, {u: w})
    with pytest.raises(ValueError, match="grouping"):
        rebind(f, {(u, v): w})


def test_conjugated_entry_and_space_validation():
    C = space("complex-binding", scalar_type="complex")
    c, d = function(C, "c"), function(C, "d")
    f = form(conjugate(c), (conjugate(c),))
    assert partial(f, {conjugate(c): d}) is conjugate(conjugate(d))
    assert rebind(f, {conjugate(c): conjugate(d)}) is form(conjugate(d), (conjugate(d),))
    with pytest.raises(ValueError, match="conjugation"):
        rebind(f, {conjugate(c): d})
    with pytest.raises(ValueError, match="spaces"):
        rebind(form(u, (u,)), {u: c})


def test_dual_argument_is_one_slot():
    load = function(dual_space(V), "load")
    pairing = form(Action(load, 0, v), (load, v))
    symbolic_load = form(u * w, (w,))
    result = partial(pairing, {load: symbolic_load})
    assert apply_actions(result) is form(u * v, (v,))


def test_alpha_comparison_is_lexical_and_does_not_change_identity():
    f, g = form(u, (u,)), form(v, (v,))
    assert f != g
    assert alpha_equivalent(f, g)
    assert not alpha_equivalent(form(form(u, (v,)), (u,)), form(form(u, (u,)), (v,)))
    assert not alpha_equivalent(form(u * v, (u, v)), form(u * v, (v, u)))
    assert not alpha_equivalent(form(u + v, ((u, v),)), form(u + v, (u, v)))


def test_derivative_declaration_is_bound_but_point_and_direction_are_free():
    derivative = directional_derivative(u * u, u, v, w)
    assert not occurs(u, derivative)
    assert occurs(v, derivative)
    assert occurs(w, derivative)
    assert alpha_equivalent(derivative, directional_derivative(p * p, p, v, w))
    assert substitute(derivative, {u: p}) is derivative


def geometry_data():
    P = RD(1)
    W = space("geometry-binding", value_shape=(1,), domain=P, scalar_type="real")
    x, y = function(W, "x"), function(W, "y")
    return x, y, pull_back(num(1), domain_map(P, x, P))


def test_geometry_occurrence_and_capture():
    x, y, e = geometry_data()
    assert occurs(x, e)
    assert occurs_in_metadata(x, e)
    assert not occurs_in_values(x, e)
    with pytest.raises(ValueError, match="capture"):
        rebind(form(e, (y,)), {y: x})
    replacement = form(integral(e, RD(1)), (y,))
    dual = function(dual_space(y.space), "dual")
    # A higher-order argument can carry the retained name only in its map data.
    with pytest.raises(ValueError, match="retained argument"):
        partial(form(Action(dual, 0, y), (dual, x)), {dual: replacement})


def test_substitution_freshens_against_metadata_capture():
    x, _, e = geometry_data()
    original = form(u, (x,))
    result = substitute(original, {u: e})
    assert result.arguments != (x,)
    assert occurs(x, result)


def test_identity_labels_do_not_create_free_occurrences():
    labelled = function(V, ("derived", u))
    assert not occurs(u, labelled)
    assert not occurs_in_metadata(u, labelled)
