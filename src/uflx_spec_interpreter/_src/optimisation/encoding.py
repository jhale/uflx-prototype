# SPDX-License-Identifier: MIT
"""Translate scalar regions, retaining opaque UFLx nodes and index identities."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

from egglog import BigInt, i64
from egglog.deconstruct import get_callable_args

from ..algorithms.alpha import AlphaHasher
from ..algorithms.binding import alpha_equivalent
from ..literals import num
from ..nodes import (
    AbstractExpression,
    Index,
    Indexed,
    IndexSum,
    Integer,
    Product,
    Sum,
    Tensor,
    Zero,
)
from ..tensors import broadcast_to, identity, index_sum, indexed
from .language import Scalar


@dataclass(frozen=True)
class EncodedRegion:
    """An egglog expression and immutable tables for facts and reconstruction."""

    expression: Scalar
    originals: tuple[AbstractExpression, ...]
    indices: tuple[Index, ...]
    terms: tuple[tuple[AbstractExpression, Scalar], ...]
    alpha_merges: int


def encode(expression: AbstractExpression, *, encode_indices: bool) -> EncodedRegion:
    """Encode one region, interning opaque atoms modulo alpha-equivalence.

    All mutable tables belong to this call. Free identities remain significant;
    alpha comparison renames only names bound inside an atom.
    """
    hasher = AlphaHasher()
    buckets: dict[int, list[int]] = {}
    atoms: dict[AbstractExpression, int] = {}
    originals: list[AbstractExpression] = []
    terms: dict[AbstractExpression, Scalar] = {}
    indices: dict[Index, int] = {}

    def index_key(index: Index) -> int:
        return indices.setdefault(index, len(indices))

    def atom_key(node: AbstractExpression) -> int:
        if node not in atoms:
            bucket = buckets.setdefault(hasher.fingerprint(node), [])
            key = next((k for k in bucket if alpha_equivalent(node, originals[k])), None)
            if key is None:
                key = len(originals)
                originals.append(node)
                bucket.append(key)
            atoms[node] = key
        return atoms[node]

    def visit(node: AbstractExpression) -> Scalar:
        if node not in terms:
            terms[node] = encode_node(node)
        return terms[node]

    def encode_node(expression: AbstractExpression) -> Scalar:
        if expression.value_shape or expression.argument_spaces:
            raise ValueError("indexed codec requires scalar values with no form arguments")
        match expression:
            case Zero():
                return Scalar.integer(BigInt(0))
            case Integer(value=value):
                return Scalar.integer(BigInt.from_string(str(value)))
            case Sum(first=a, second=b):
                return visit(a) + visit(b)
            case Product(first=a, second=b):
                return visit(a) * visit(b)
            case IndexSum(expression=body, indices=(i,)) if encode_indices and isinstance(
                n := dict(body.free_indices)[i], int
            ):
                return Scalar.summed(visit(body), i64(index_key(i)), i64(n))
            case Indexed(expression=Tensor() as source, indices=(Index() as i, Index() as j)) if (
                encode_indices
                and isinstance(n := source.value_shape[0], int)
                and source.value_shape == (n, n)
                and source == identity(n)
            ):
                return Scalar.delta(i64(n), i64(index_key(i)), i64(index_key(j)))
            case Indexed(expression=source, indices=(Index() as i,)) if (
                encode_indices
                and not source.free_indices
                and isinstance(n := source.value_shape[0], int)
            ):
                return (
                    Scalar.integer(BigInt(0))
                    if isinstance(source, Zero)
                    else Scalar.at(i64(atom_key(source)), i64(index_key(i)), i64(n))
                )
            case _:
                return Scalar.atom(i64(atom_key(expression)))

    encoded = visit(expression)
    return EncodedRegion(
        encoded, tuple(originals), tuple(indices), tuple(terms.items()), len(atoms) - len(originals)
    )


def decode(expression: Scalar, region: EncodedRegion) -> AbstractExpression:
    """Reconstruct an expression using the original opaque objects."""

    @cache
    def visit(expression: Scalar) -> AbstractExpression:
        if (args := get_callable_args(expression, Scalar.at)) is not None:
            k, i, _ = (a.value for a in args)
            return indexed(region.originals[k], (region.indices[i],))
        if (args := get_callable_args(expression, Scalar.delta)) is not None:
            n, i, j = (a.value for a in args)
            return indexed(identity(n), (region.indices[i], region.indices[j]))
        if (args := get_callable_args(expression, Scalar.summed)) is not None:
            child, i, n = args
            body = visit(child)
            index, extent = region.indices[i.value], n.value
            if index not in dict(body.free_indices):
                body = indexed(broadcast_to(body, (extent,)), (index,))
            return index_sum(body, (index,))
        if (args := get_callable_args(expression, Scalar.integer)) is not None:
            return num(args[0].value)
        if (args := get_callable_args(expression, Scalar.atom)) is not None:
            return region.originals[args[0].value]
        if (args := get_callable_args(expression, Scalar.__add__)) is not None:
            return visit(args[0]) + visit(args[1])
        if (args := get_callable_args(expression, Scalar.__mul__)) is not None:
            return visit(args[0]) * visit(args[1])
        raise TypeError(f"unexpected extracted expression: {expression}")

    return visit(expression)
