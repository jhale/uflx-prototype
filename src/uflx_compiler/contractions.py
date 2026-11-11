"""Collect scalar contractions by their output-index dependencies."""

from functools import cache
from itertools import product
from math import prod

import numpy as np

from uflx_spec_interpreter import add, divide, index_sum, multiply, num
from uflx_spec_interpreter._src.algorithms.rewriting import complete_indices
from uflx_spec_interpreter._src.algorithms.substitution import substitute
from uflx_spec_interpreter._src.algorithms.traversal import map_values, memoized
from uflx_spec_interpreter._src.nodes import (
    Division,
    Index,
    Indexed,
    IndexSum,
    Product,
    Sum,
    Tensor,
    Zero,
)
from uflx_spec_interpreter._src.nodes.structure import expression_operands

from .algebra import _normalizer
from .evaluation import evaluate
from .inputs import arrays
from .quadrature import constant
from .simplification import literal, simplify


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


def _preintegrate(
    expression, output_indices, *, share_tables=False, max_terms=64, max_entries=65536
):
    """Tabulate reference contractions whose runtime factors are reduction-invariant.

    This is mathematical reassociation, including compile-time floating-point
    summation. Expansion and table sizes are bounded; unsupported or varying
    runtime factors leave the original reduction intact. Shared tables keep scalar
    scales outside the reference factors, allowing reuse across component blocks.
    """
    reads = memoized(arrays)
    normalize = _normalizer()

    @cache
    def selected_indices(node):
        own = (
            {i for i in node.indices if isinstance(i, Index)}
            if isinstance(node, Indexed) and isinstance(node.expression, Tensor)
            else set()
        )
        return own.union(*(selected_indices(c) for c in expression_operands(node)))

    class Unsupported(Exception):
        pass

    @cache
    def tabulate(reference):
        reference = normalize(reference)
        free_indices = tuple(i for i, _ in reference.free_indices)
        indices = tuple(i for i in output_indices if i in free_indices) + tuple(
            i for i in free_indices if i not in output_indices
        )
        data = evaluate(reference, {})
        data = np.transpose(data, tuple(free_indices.index(i) for i in indices))
        values = constant(data)
        return Indexed(values, indices) if indices else values

    def integrate(node):
        axes = set(node.indices)
        extents = dict(node.expression.free_indices)

        @cache
        def terms(n):
            # Keep scalar scales in the coefficient, rather than creating scaled
            # copies of a reference table. Split constant products for the same reason.
            if share_tables and (value := literal(n)) is not None:
                return ((num(value), num(1)),)
            if share_tables and not n.free_indices and not n.value_shape:
                return ((n, num(1)),)
            if not reads(n) and (not share_tables or not isinstance(n, (Product, Division))):
                return ((num(1), n),)
            if not axes.intersection(i for i, _ in n.free_indices):
                return ((n, num(1)),)
            if isinstance(n, Sum):
                result = terms(n.first) + terms(n.second)
            elif isinstance(n, Product):
                left, right = terms(n.first), terms(n.second)
                if len(left) * len(right) > max_terms:
                    raise Unsupported
                result = tuple((multiply(a, b), multiply(x, y)) for a, x in left for b, y in right)
            elif isinstance(n, Division):
                denominator = n.second
                if (share_tables and not denominator.free_indices) or (
                    reads(denominator)
                    and not axes.intersection(i for i, _ in denominator.free_indices)
                ):
                    result = tuple((divide(a, denominator), x) for a, x in terms(n.first))
                elif not reads(denominator):
                    result = tuple((a, divide(x, denominator)) for a, x in terms(n.first))
                else:
                    raise Unsupported
            elif isinstance(n, IndexSum) and (
                not axes.intersection(n.indices)
                and prod(dict(n.expression.free_indices)[i] for i in n.indices) <= max_terms
                and (
                    len(output_indices) == 1
                    or all(
                        dict(n.expression.free_indices)[i] <= 3
                        and i in selected_indices(n.expression)
                        for i in n.indices
                    )
                )
            ):
                # Actions may expose coefficient DOFs as well as spatial components.
                # The same term budget bounds products of these expansions.
                sizes = dict(n.expression.free_indices)
                result = ()
                for values in product(*(range(sizes[i]) for i in n.indices)):
                    entry = simplify(
                        substitute(n.expression, dict(zip(n.indices, values, strict=True)))
                    )
                    result += terms(entry)
                    if len(result) > max_terms:
                        raise Unsupported
            else:
                raise Unsupported
            if len(result) > max_terms:
                raise Unsupported
            return result

        try:
            separated = terms(node.expression)
            if not share_tables:
                # A single output block needs fewer runtime products when equal
                # coefficients are combined before reference integration.
                collected = {}
                for coefficient, reference in separated:
                    coefficient = normalize(simplify(coefficient))
                    collected[coefficient] = (
                        add(collected[coefficient], reference)
                        if coefficient in collected
                        else reference
                    )
                separated = tuple(collected.items())
            tables = {}
            entries = 0
            for coefficient, reference in separated:
                coefficient = normalize(simplify(coefficient))
                free = dict(reference.free_indices)
                bound = tuple(i for i in node.indices if i in free)
                reference = index_sum(reference, bound) if bound else reference
                # A summand may be independent of one or more reduction indices.
                reference = multiply(
                    num(prod(extents[i] for i in node.indices if i not in free)), reference
                )
                entries += prod(size for _, size in reference.free_indices)
                if entries > max_entries:
                    raise Unsupported
                table = tabulate(reference)
                tables[table] = add(tables[table], coefficient) if table in tables else coefficient
            return complete_indices(
                node,
                simplify(
                    _sum(
                        multiply(normalize(c) if share_tables else c, t) for t, c in tables.items()
                    )
                ),
            )
        except Unsupported:
            return node

    def walk(walk, node):
        node = map_values(node, walk)
        if (
            isinstance(node, IndexSum)
            and not node.value_shape
            and set(output_indices) <= dict(node.free_indices).keys()
            and prod(size for _, size in node.expression.free_indices) <= max_entries
        ):
            return integrate(node)
        return node

    return memoized(walk)(expression)
