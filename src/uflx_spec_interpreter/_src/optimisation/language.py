# SPDX-License-Identifier: MIT
"""Internal egglog language for exact scalar algebra and limited index sums."""

from __future__ import annotations

from egglog import BigInt, Expr, i64


class Scalar(Expr):
    """Exact scalar algebra; opaque atoms may be mathematically undefined."""

    @classmethod
    def integer(cls, value: BigInt) -> Scalar: ...

    @classmethod
    def atom(cls, key: i64) -> Scalar: ...

    @classmethod
    def at(cls, key: i64, index: i64, extent: i64) -> Scalar: ...

    @classmethod
    def delta(cls, extent: i64, first: i64, second: i64) -> Scalar: ...

    @classmethod
    def summed(cls, body: Scalar, index: i64, extent: i64) -> Scalar: ...

    def __add__(self, other: Scalar) -> Scalar: ...

    def __mul__(self, other: Scalar) -> Scalar: ...
