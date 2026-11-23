"""The same lexical laws for form, tensor, sum and derivative binders."""

import pytest

from uflx._src.algorithms.binding import alpha_equivalent, binding_signature
from uflx._src.algorithms.substitution import occurs, rename_bound, substitute
from uflx.arithmetic import num
from uflx.calculus import directional_derivative
from uflx.forms import form, function, rebind
from uflx.tensors import component_tensor, index, index_sum, tensor

from .spaces import space

V = space("shared-binding", scalar_type="real")
u, v, w, a = (function(V, name) for name in ("u", "v", "w", "a"))
i, j, k = (index(name) for name in ("i", "j", "k"))
vector = tensor((num(2), num(3)))


@pytest.fixture(params=["form", "tensor", "sum", "derivative"])
def binder(request):
    match request.param:
        case "form":
            return form(u * w, (u,)), u, v
        case "tensor":
            return component_tensor(vector[i] * w, (i,)), i, j
        case "sum":
            return index_sum(vector[i] * w, (i,)), i, j
        case "derivative":
            return directional_derivative(u * w, u, a, a), u, v


def test_renaming_preserves_signature_and_alpha_equivalence(binder):
    node, old, new = binder
    renamed = rename_bound(node, {old: new})
    assert renamed is not node
    assert renamed.bound() == (new,)
    assert binding_signature(renamed) == binding_signature(node)
    assert alpha_equivalent(node, renamed)
    assert not occurs(old, node)
    assert not occurs(new, renamed)
    assert occurs(w, node)
    assert rename_bound(renamed, {new: old}) is node


def test_substitution_does_not_replace_declarations(binder):
    node, old, new = binder
    assert substitute(node, {old: new}) is node


def test_capture_avoidance_preserves_inserted_free_names(binder):
    node, old, new = binder
    inserted = old if old is u else vector[i]
    result = substitute(node, {w: inserted})
    expected = substitute(rename_bound(node, {old: new}), {w: inserted})
    assert result.bound() != (old,)
    assert occurs(old, result)
    assert alpha_equivalent(result, expected)
    assert substitute(node, {w: inserted}) is result


def test_unknown_declarations_are_rejected(binder):
    node, _, new = binder
    with pytest.raises(ValueError, match="unknown bound"):
        rename_bound(node, {new: new})


@pytest.mark.parametrize("constructor", [component_tensor, index_sum])
def test_public_rebind_handles_index_binders(constructor):
    node = constructor(vector[i] * vector[j], (i,))
    assert rebind(node, {i: k}) is constructor(vector[k] * vector[j], (k,))
    with pytest.raises(ValueError, match="capture"):
        rebind(node, {i: j})


def test_index_swaps_preserve_ordered_extents():
    matrix = tensor((tensor((num(1), num(2), num(3))), tensor((num(4), num(5), num(6)))))
    node = component_tensor(matrix[i, j], (i, j))
    result = rebind(node, {i: j, j: i})
    assert result is component_tensor(matrix[j, i], (j, i))
    assert binding_signature(result) == (2, 3)
    assert alpha_equivalent(node, result)
    with pytest.raises(ValueError, match="duplicate"):
        rebind(node, {i: j})


def test_derivative_point_and_direction_are_outside_the_renamed_scope():
    node = directional_derivative(u * u, u, v, v)
    # v is free outside the scope: renaming u to v does not capture that use.
    renamed = rename_bound(node, {u: v})
    assert renamed is directional_derivative(v * v, v, v, v)
    assert alpha_equivalent(node, renamed)
    assert occurs(v, renamed)
    assert substitute(renamed, {v: w}) is directional_derivative(v * v, v, w, w)


def test_renaming_tuple_derivative_preserves_variable_order_and_evaluation_operands():
    node = directional_derivative(u * v, (u, v), (v, u), (w, a))
    assert rename_bound(node, {v: u, u: v}) is directional_derivative(v * u, (v, u), (v, u), (w, a))
    with pytest.raises(TypeError, match="explicit Form"):
        rebind(node, {u: w})


def test_derivative_body_capture_is_rejected():
    node = directional_derivative(u * v, u, a, a)
    with pytest.raises(ValueError, match="capture"):
        rename_bound(node, {u: v})


def test_function_and_index_scopes_nest_without_a_special_rule():
    node = form(index_sum(u * vector[i], (i,)), (u,))
    renamed = form(index_sum(v * vector[j], (j,)), (v,))
    assert alpha_equivalent(node, renamed)
    inserted = substitute(node, {vector: tensor((u, w))})
    assert occurs(u, inserted)
    assert inserted.bound() != (u,)


def test_alpha_comparison_distinguishes_shadowing_of_shared_leaves():
    left = form(form(u, (v,)), (u,))
    right = form(form(u, (u,)), (v,))
    assert not alpha_equivalent(left, right)
    left = component_tensor(component_tensor(vector[i] * vector[j], (j,))[0], (i,))
    right = component_tensor(component_tensor(vector[i] * vector[j], (i,))[0], (j,))
    assert not alpha_equivalent(left, right)


def test_alpha_comparison_preserves_signatures_and_free_identity():
    other = function(space("other-binding", scalar_type="real"), "v")
    assert not alpha_equivalent(form(u, (u,)), form(other, (other,)))
    assert not alpha_equivalent(form(u * w, (u,)), form(v * a, (v,)))
    assert not alpha_equivalent(
        index_sum(vector[i], (i,)), index_sum(tensor((num(2), num(3), num(4)))[j], (j,))
    )


def test_alpha_comparison_handles_a_shared_dag():
    left, right = u, v
    for _ in range(24):
        left, right = left + left, right + right
    assert alpha_equivalent(form(left, (u,)), form(right, (v,)))
