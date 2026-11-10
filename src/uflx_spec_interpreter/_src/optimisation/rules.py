# SPDX-License-Identifier: MIT
"""Algebraic identities, guarded index rules and non-expanding cleanup."""

from collections.abc import Iterator

from egglog import BigRat, RewriteOrRule, birewrite, eq, i64, ne, relation, rewrite, rule, ruleset

from .language import Scalar


@ruleset
def algebra(a: Scalar, b: Scalar, c: Scalar) -> Iterator[RewriteOrRule]:
    yield rewrite(a + b).to(b + a)
    yield rewrite(a * b).to(b * a)
    yield rewrite((a + b) + c).to(a + (b + c))
    yield rewrite((a * b) * c).to(a * (b * c))
    yield birewrite(a * (b + c)).to(a * b + a * c)
    yield rewrite(a + a).to(Scalar.number(BigRat(2, 1)) * a)


independent = relation("independent", i64, i64)
defined = relation("defined", Scalar)
is_real = relation("is_real", Scalar)


@ruleset
def unary_rules(a: Scalar) -> Iterator[RewriteOrRule]:
    """Unary identities that preserve evaluation of their remaining operands."""
    yield rewrite(a.absolute().absolute()).to(a.absolute())
    yield rewrite(a.conjugate().conjugate()).to(a)
    yield rewrite(a.real_part()).to(a, is_real(a))
    yield rewrite(a.imaginary_part()).to(Scalar.number(BigRat(0, 1)), is_real(a), defined(a))


@ruleset
def indexed_rules(a: Scalar, k: i64, i: i64, j: i64, n: i64) -> Iterator[RewriteOrRule]:
    zero = Scalar.number(BigRat(0, 1))
    yield rewrite(Scalar.summed(a * Scalar.atom(k), j, n)).to(
        Scalar.summed(a, j, n) * Scalar.atom(k), independent(k, j), n > 0
    )
    yield rewrite(Scalar.summed(zero, j, n)).to(zero)
    yield rewrite(Scalar.delta(n, i, j)).to(Scalar.delta(n, j, i))
    # The vector is independent of both indices; n must match in all three terms.
    yield rewrite(Scalar.summed(Scalar.delta(n, i, j) * Scalar.at(k, j, n), j, n)).to(
        Scalar.at(k, i, n), ne(i).to(j), defined(Scalar.at(k, j, n))
    )


@ruleset
def constant_cleanup(a: Scalar, i: BigRat, j: BigRat) -> Iterator[RewriteOrRule]:
    """Non-expanding constant rules shared by search and finishing cleanup."""
    zero, one = Scalar.number(BigRat(0, 1)), Scalar.number(BigRat(1, 1))
    yield rewrite(Scalar.number(i) + Scalar.number(j)).to(Scalar.number(i + j))
    yield rewrite(Scalar.number(i) * Scalar.number(j)).to(Scalar.number(i * j))
    yield rule(eq(a).to(Scalar.number(i))).then(defined(a))
    yield rewrite(a / Scalar.number(j)).to(
        a * Scalar.number(BigRat(1, 1) / j), ne(j).to(BigRat(0, 1))
    )
    yield rewrite(a + zero).to(a)
    yield rewrite(zero + a).to(a)
    yield rewrite(a * zero).to(zero, defined(a))
    yield rewrite(zero * a).to(zero, defined(a))
    yield rewrite(a * one).to(a)
    yield rewrite(one * a).to(a)
    yield rewrite(a.power(one)).to(a)
    yield rewrite(a.power(zero)).to(one, defined(a))
    yield rewrite(one.logarithm()).to(zero)
