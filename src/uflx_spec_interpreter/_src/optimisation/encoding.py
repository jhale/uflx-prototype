# SPDX-License-Identifier: MIT
"""Translate scalar regions, retaining opaque UFLx nodes and index identities."""

from __future__ import annotations

from typing import cast

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


class Codec:
    """Allow free indices, with an explicit, deliberately small indexed language."""

    def __init__(self, encode_indices: bool) -> None:
        self.encode_indices = encode_indices
        self.hasher = AlphaHasher()
        self.buckets: dict[int, list[int]] = {}
        self.alpha_merges = 0
        self.atoms: dict[AbstractExpression, int] = {}
        self.originals: list[AbstractExpression] = []
        self.memo: dict[AbstractExpression, Scalar] = {}
        self.indices: dict[Index, int] = {}
        self.index_nodes: list[Index] = []

    def index_key(self, index: Index) -> int:
        if index not in self.indices:
            self.indices[index] = len(self.index_nodes)
            self.index_nodes.append(index)
        return self.indices[index]

    def atom_key(self, expression: AbstractExpression) -> int:
        """Intern modulo alpha-equivalence, preserving all free identities.

        A representative may change names bound inside this atom only. No enclosing
        binding environment is supplied, so open bodies cannot be identified by
        renaming their free names.
        """
        if expression in self.atoms:
            return self.atoms[expression]
        fingerprint = self.hasher.fingerprint(expression)
        bucket = self.buckets.setdefault(fingerprint, [])
        for key in bucket:
            if alpha_equivalent(expression, self.originals[key]):
                self.atoms[expression] = key
                self.alpha_merges += 1
                return key
        key = len(self.originals)
        self.originals.append(expression)
        bucket.append(key)
        self.atoms[expression] = key
        return key

    def encode(self, expression: AbstractExpression) -> Scalar:
        if expression in self.memo:
            return self.memo[expression]
        if expression.value_shape or expression.argument_spaces:
            raise ValueError("indexed codec requires scalar values with no form arguments")
        if isinstance(expression, Zero):
            result = Scalar.integer(BigInt(0))
        elif isinstance(expression, Integer):
            result = Scalar.integer(BigInt.from_string(str(expression.value)))
        elif isinstance(expression, Sum | Product):
            a, b = self.encode(expression.first), self.encode(expression.second)
            result = a + b if isinstance(expression, Sum) else a * b
        elif (
            self.encode_indices
            and isinstance(expression, IndexSum)
            and len(expression.indices) == 1
            and isinstance(
                n := dict(expression.expression.free_indices)[expression.indices[0]], int
            )
        ):
            result = Scalar.summed(
                self.encode(expression.expression),
                i64(self.index_key(expression.indices[0])),
                i64(n),
            )
        elif (
            self.encode_indices
            and isinstance(expression, Indexed)
            and len(expression.indices) == 2
            and all(isinstance(i, Index) for i in expression.indices)
            and isinstance(expression.expression, Tensor)
            and len(shape := expression.expression.value_shape) == 2
            and isinstance(shape[0], int)
            and shape[0] == shape[1]
            and expression.expression == identity(shape[0])
        ):
            result = Scalar.delta(
                i64(shape[0]), *(i64(self.index_key(cast(Index, i))) for i in expression.indices)
            )
        elif (
            self.encode_indices
            and isinstance(expression, Indexed)
            and len(expression.indices) == 1
            and isinstance(expression.indices[0], Index)
            and not expression.expression.free_indices
            and len(shape := expression.expression.value_shape) == 1
            and isinstance(shape[0], int)
        ):
            result = (
                Scalar.integer(BigInt(0))
                if isinstance(expression.expression, Zero)
                else Scalar.at(
                    i64(self.atom_key(expression.expression)),
                    i64(self.index_key(expression.indices[0])),
                    i64(shape[0]),
                )
            )
        else:
            result = Scalar.atom(i64(self.atom_key(expression)))
        self.memo[expression] = result
        return result

    def decode(self, expression: Scalar) -> AbstractExpression:
        if (args := get_callable_args(expression, Scalar.at)) is not None:
            k, i, _ = (a.value for a in args)
            return indexed(self.originals[k], (self.index_nodes[i],))
        if (args := get_callable_args(expression, Scalar.delta)) is not None:
            n, i, j = (a.value for a in args)
            return indexed(identity(n), (self.index_nodes[i], self.index_nodes[j]))
        if (args := get_callable_args(expression, Scalar.summed)) is not None:
            child, i, n = args
            body = self.decode(child)
            index, extent = self.index_nodes[i.value], n.value
            if index not in dict(body.free_indices):
                body = indexed(broadcast_to(body, (extent,)), (index,))
            return index_sum(body, (index,))
        if (args := get_callable_args(expression, Scalar.integer)) is not None:
            return num(args[0].value)
        if (args := get_callable_args(expression, Scalar.atom)) is not None:
            return self.originals[args[0].value]
        if (args := get_callable_args(expression, Scalar.__add__)) is not None:
            return self.decode(args[0]) + self.decode(args[1])
        if (args := get_callable_args(expression, Scalar.__mul__)) is not None:
            return self.decode(args[0]) * self.decode(args[1])
        raise TypeError(f"unexpected extracted expression: {expression}")
