"""Cancel contracted components of a full-column-rank cell Jacobian and its inverse."""

from functools import cache, reduce
from operator import mul

from uflx.algorithms import bottom_up, complete_indices, derived_index, substitute_values
from uflx.arithmetic import num
from uflx.nodes import ComponentTensor, Division, Index, Indexed, IndexSum, Inverse, Product
from uflx.tensors import identity, index_sum


def _contract_entries(a, b, axis, jacobians):
    """Return the identity entry produced by summing two matrix entries."""
    if not isinstance(a, Indexed) or not isinstance(b, Indexed):
        return None
    if isinstance(a.expression, Inverse):
        a, b = b, a
    J = a.expression
    if J not in jacobians or not isinstance(b.expression, Inverse) or b.expression.matrix != J:
        return None
    # K J is the reference identity. J K is a projector on embedded cells.
    if a.indices[0] == axis and b.indices[1] == axis:
        dimension = J.value_shape[1]
        left, right = b.indices[0], a.indices[1]
    elif a.indices[1] == axis and b.indices[0] == axis and J.value_shape[0] == J.value_shape[1]:
        dimension = J.value_shape[0]
        left, right = a.indices[0], b.indices[1]
    else:
        return None
    if axis in (left, right) or not all(isinstance(i, (Index, int)) for i in (left, right)):
        return None
    return dimension, left, right


def _cancel_sum(node, jacobians, geometry):
    # Open only the multiplicative part of this contraction. Keep the original
    # graph unless a pair cancels. Fresh indices keep independent sums disjoint.
    axes = list(node.indices)
    factors = []
    denominators = []
    extents = dict(node.expression.free_indices)

    def collect(term):
        if not geometry(term) or dict(term.free_indices).keys().isdisjoint(axes):
            factors.append(term)
            return
        match term:
            case Indexed(ComponentTensor(body, bound), indices):
                collect(substitute_values(body, dict(zip(bound, indices, strict=True))))
            case IndexSum(body, bound):
                fresh = [
                    derived_index("uflx_jax.contract", node, len(axes) + j)
                    for j in range(len(bound))
                ]
                sizes = dict(body.free_indices)
                extents.update((new, sizes[old]) for old, new in zip(bound, fresh, strict=True))
                axes.extend(fresh)
                collect(substitute_values(body, dict(zip(bound, fresh, strict=True))))
            case Product(a, b):
                collect(a)
                collect(b)
            case Division(a, b):
                collect(a)
                denominators.append(b)
            case _:
                factors.append(term)

    collect(node.expression)
    if any(not isinstance(extents[i], int) or extents[i] < 1 for i in axes):
        return node
    changed = False
    for axis in tuple(axes):
        if axis not in axes or any(axis in dict(d.free_indices) for d in denominators):
            continue
        uses = [i for i, term in enumerate(factors) if axis in dict(term.free_indices)]
        if len(uses) != 2:
            continue
        a, b = (factors[i] for i in uses)
        delta = _contract_entries(a, b, axis, jacobians)
        if delta is None:
            continue
        dimension, left, right = delta
        changed = True
        axes.remove(axis)
        factors = [term for i, term in enumerate(factors) if i not in uses]
        # In sum_ij delta_ij f(i) g(j), both operands still visit every
        # component after merging the axes. Keep the delta for a joint f(i,j):
        # selecting only its diagonal could discard undefined off-diagonal values.
        if (
            left != right
            and left in axes
            and right in axes
            and all(
                not {left, right} <= dict(term.free_indices).keys()
                for term in (*factors, *denominators)
            )
        ):
            axes.remove(left)
            factors = [substitute_values(term, {left: right}) for term in factors]
            denominators = [substitute_values(term, {left: right}) for term in denominators]
        else:
            factors.append(identity(dimension)[left, right])

    if not changed:
        return node
    result = reduce(mul, factors) if factors else num(1)
    for denominator in denominators:
        result = result / denominator
    # A sum of a constant still contributes its extent.
    for axis in axes:
        if axis not in dict(result.free_indices):
            result = result * num(extents[axis])
    bound = tuple(i for i in axes if i in dict(result.free_indices))
    return complete_indices(node, index_sum(result, bound))


def cancel_jacobian_products(expression, jacobians):
    """Cancel scalar index contractions for the supplied cell-map Jacobians."""
    if not jacobians:
        return expression

    @cache
    def geometry(node):
        # Only multiplicative paths can expose a cancellable pair. Treat, for
        # example, a determinant or a sum of material terms as an opaque factor.
        if node in jacobians:
            return 1
        match node:
            case Inverse(matrix):
                return 2 if matrix in jacobians else 0
            case Product(a, b):
                return geometry(a) | geometry(b)
            case Division(a, _) | Indexed(a, _) | ComponentTensor(a, _) | IndexSum(a, _):
                return geometry(a)
        return 0

    def cancel(node):
        if isinstance(node, IndexSum) and not node.argument_spaces and geometry(node) == 3:
            return _cancel_sum(node, jacobians, geometry)
        return node

    return bottom_up(expression, cancel)
