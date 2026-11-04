"""Collect scalar contractions by their output-index dependencies."""

from functools import cache
from itertools import product

from uflx_spec_interpreter import add, multiply, num
from uflx_spec_interpreter._src.algorithms.substitution import substitute
from uflx_spec_interpreter._src.nodes import Index, Indexed, IndexSum, Product, Sum, Tensor, Zero
from uflx_spec_interpreter._src.nodes.structure import expression_operands

from .algebra import _normalizer
from .simplification import simplify


def _factor_contraction(expression, rows, outer_indices, *, combine_geometry):
    """Prepare coefficients of repeated trial factors outside the basis-pair loop.

    Only distribute products with one index-dependent operand. Products of two
    dependent expressions remain intact, avoiding polynomial expansion. Small
    tensor-component sums are expanded when their transformation coefficients
    are invariant over the outer integration loops.
    """
    test, trial = rows

    @cache
    def selected_indices(expression):
        """Indices selecting explicit tensor components, rather than general reductions."""
        indices = set()
        if isinstance(expression, Indexed) and isinstance(expression.expression, Tensor):
            indices.update(i for i in expression.indices if isinstance(i, Index))
        for child in expression_operands(expression):
            indices.update(selected_indices(child))
        return frozenset(indices)

    normalize = _normalizer()

    def collect(expression, index, expand_components=True):
        @cache
        def terms(node):
            if index not in dict(node.free_indices):
                return {None: node}
            if combine_geometry and expand_components and isinstance(node, IndexSum):
                extents = dict(node.expression.free_indices)
                if all(
                    extents[i] <= 3 and i in selected_indices(node.expression) for i in node.indices
                ):
                    entries = (
                        simplify(
                            substitute(
                                node.expression, dict(zip(node.indices, values, strict=True))
                            )
                        )
                        for values in product(*(range(extents[i]) for i in node.indices))
                    )
                    expanded = terms(_sum(entries))
                    if all(
                        not set(outer_indices).intersection(i for i, _ in c.free_indices)
                        for c in expanded.values()
                    ):
                        return expanded
            if isinstance(node, Sum):
                result = dict(terms(node.first))
                for factor, coefficient in terms(node.second).items():
                    result[factor] = (
                        add(result[factor], coefficient) if factor in result else coefficient
                    )
                return result
            if isinstance(node, Product):
                left, right = node.first, node.second
                if index not in dict(right.free_indices):
                    left, right = right, left
                if index not in dict(left.free_indices):
                    return {factor: multiply(left, c) for factor, c in terms(right).items()}
            return {node: num(1)}

        return terms(expression)

    def combine(terms):
        parts = [c if factor is None else multiply(c, factor) for factor, c in terms.items()]
        result = parts[0]
        for part in parts[1:]:
            result = add(result, part)
        return result

    trial_terms = collect(expression, trial, expand_components=False)
    # Keep invariant basis contractions intact: distributing a varying weight
    # into them can prevent the C compiler from hoisting their arithmetic.
    if outer_indices and not any(
        set(outer_indices).intersection(i for i, _ in factor.free_indices)
        for factor in trial_terms
        if factor is not None and not isinstance(factor, Zero)
    ):
        return expression
    expanded = collect(expression, trial)
    changed = expanded.keys() != trial_terms.keys()
    terms = {}
    for factor, coefficient in expanded.items():
        coefficients = collect(coefficient, test, expand_components=changed)
        if changed:
            coefficients = {
                basis: normalize(_common_factors(simplify(c))) for basis, c in coefficients.items()
            }
        terms[factor] = combine(coefficients)
    return simplify(combine(terms))


def _sum(entries):
    result = num(0)
    for entry in entries:
        result = add(result, entry)
    return result


def _product(entries):
    result = num(1)
    for entry in entries:
        result = multiply(result, entry)
    return result


def _common_factors(expression):
    """Pull shared weights out of sums of geometry products before scheduling."""

    def flatten(node, operator):
        if isinstance(node, operator):
            return flatten(node.first, operator) + flatten(node.second, operator)
        return [node]

    terms = [flatten(term, Product) for term in flatten(expression, Sum)]
    if len(terms) < 2:
        return expression
    common = []
    for factor in tuple(terms[0]):
        if all(factor in term for term in terms):
            common.append(factor)
            for term in terms:
                term.remove(factor)
    if not common:
        return expression
    return multiply(_product(common), _sum(_product(term) for term in terms))
