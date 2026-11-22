# Copyright (c) 2015-2024 Herbie Project
# Modified work Copyright 2016 Google Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# SPDX-License-Identifier: MIT
# Generated from Herbie 81fe5efeb5accea858051e5a53264fdc925f9a65; do not edit.
# Regenerate: python3 tools/translate_herbie.py
"""Opt-in Herbie identities translated to UFLx scalar terms."""

from collections.abc import Iterator

from egglog import BigRat, RewriteOrRule, rewrite, ruleset

from .language import Scalar
from .rules import defined


@ruleset
def herbie_arithmetic(
    a: Scalar, b: Scalar, c: Scalar, d: Scalar, x: Scalar, y: Scalar
) -> Iterator[RewriteOrRule]:
    """Translated arithmetic rules with definedness guards."""
    # Herbie: +-commutative
    yield rewrite(a + b).to(b + a)
    # Herbie: *-commutative
    yield rewrite(a * b).to(b * a)
    # Herbie: associate-+r+
    yield rewrite(a + (b + c)).to((a + b) + c)
    # Herbie: associate-+l+
    yield rewrite((a + b) + c).to(a + (b + c))
    # Herbie: associate-+r-
    yield rewrite(a + (b + (Scalar.number(BigRat(-1, 1)) * c))).to(
        (a + b) + (Scalar.number(BigRat(-1, 1)) * c)
    )
    # Herbie: associate-+l-
    yield rewrite((a + (Scalar.number(BigRat(-1, 1)) * b)) + c).to(
        a + (Scalar.number(BigRat(-1, 1)) * (b + (Scalar.number(BigRat(-1, 1)) * c)))
    )
    # Herbie: associate--r+
    yield rewrite(a + (Scalar.number(BigRat(-1, 1)) * (b + c))).to(
        (a + (Scalar.number(BigRat(-1, 1)) * b)) + (Scalar.number(BigRat(-1, 1)) * c)
    )
    # Herbie: associate--l+
    yield rewrite((a + b) + (Scalar.number(BigRat(-1, 1)) * c)).to(
        a + (b + (Scalar.number(BigRat(-1, 1)) * c))
    )
    # Herbie: associate--l-
    yield rewrite((a + (Scalar.number(BigRat(-1, 1)) * b)) + (Scalar.number(BigRat(-1, 1)) * c)).to(
        a + (Scalar.number(BigRat(-1, 1)) * (b + c))
    )
    # Herbie: associate--r-
    yield rewrite(a + (Scalar.number(BigRat(-1, 1)) * (b + (Scalar.number(BigRat(-1, 1)) * c)))).to(
        (a + (Scalar.number(BigRat(-1, 1)) * b)) + c
    )
    # Herbie: associate-*r*
    yield rewrite(a * (b * c)).to((a * b) * c)
    # Herbie: associate-*l*
    yield rewrite((a * b) * c).to(a * (b * c))
    # Herbie: associate-*cross
    yield rewrite((a * b) * (c * d)).to((a * c) * (b * d))
    # Herbie: +-inverses
    yield rewrite(a + (Scalar.number(BigRat(-1, 1)) * a)).to(
        Scalar.number(BigRat(0, 1)), defined(a)
    )
    # Herbie: mul0-lft
    yield rewrite(Scalar.number(BigRat(0, 1)) * a).to(Scalar.number(BigRat(0, 1)), defined(a))
    # Herbie: mul0-rgt
    yield rewrite(a * Scalar.number(BigRat(0, 1))).to(Scalar.number(BigRat(0, 1)), defined(a))
    # Herbie: +-lft-identity
    yield rewrite(Scalar.number(BigRat(0, 1)) + a).to(a)
    # Herbie: +-rgt-identity
    yield rewrite(a + Scalar.number(BigRat(0, 1))).to(a)
    # Herbie: --rgt-identity
    yield rewrite(a + (Scalar.number(BigRat(-1, 1)) * Scalar.number(BigRat(0, 1)))).to(a)
    # Herbie: sub0-neg
    yield rewrite(Scalar.number(BigRat(0, 1)) + (Scalar.number(BigRat(-1, 1)) * a)).to(
        Scalar.number(BigRat(-1, 1)) * a
    )
    # Herbie: remove-double-neg
    yield rewrite(Scalar.number(BigRat(-1, 1)) * (Scalar.number(BigRat(-1, 1)) * a)).to(a)
    # Herbie: *-lft-identity
    yield rewrite(Scalar.number(BigRat(1, 1)) * a).to(a)
    # Herbie: *-rgt-identity
    yield rewrite(a * Scalar.number(BigRat(1, 1))).to(a)
    # Herbie: /-rgt-identity
    yield rewrite(a / Scalar.number(BigRat(1, 1))).to(a)
    # Herbie: count-2
    yield rewrite(x + x).to(Scalar.number(BigRat(2, 1)) * x)
    # Herbie: 2-split
    yield rewrite(Scalar.number(BigRat(2, 1))).to(
        Scalar.number(BigRat(1, 1)) + Scalar.number(BigRat(1, 1))
    )
    # Herbie: 4-split
    yield rewrite(Scalar.number(BigRat(4, 1))).to(
        Scalar.number(BigRat(2, 1)) * Scalar.number(BigRat(2, 1))
    )
    # Herbie: count-2-rev
    yield rewrite(Scalar.number(BigRat(2, 1)) * x).to(x + x)
    # Herbie: 1-split
    yield rewrite(Scalar.number(BigRat(1, 1))).to(
        Scalar.number(BigRat(2, 1)) * Scalar.number(BigRat(1, 2))
    )
    # Herbie: distribute-lft-in
    yield rewrite(a * (b + c)).to((a * b) + (a * c))
    # Herbie: distribute-rgt-in
    yield rewrite(a * (b + c)).to((b * a) + (c * a))
    # Herbie: distribute-lft-out
    yield rewrite((a * b) + (a * c)).to(a * (b + c))
    # Herbie: distribute-lft-out--
    yield rewrite((a * b) + (Scalar.number(BigRat(-1, 1)) * (a * c))).to(
        a * (b + (Scalar.number(BigRat(-1, 1)) * c))
    )
    # Herbie: distribute-rgt-out
    yield rewrite((b * a) + (c * a)).to(a * (b + c))
    # Herbie: distribute-rgt-out--
    yield rewrite((b * a) + (Scalar.number(BigRat(-1, 1)) * (c * a))).to(
        a * (b + (Scalar.number(BigRat(-1, 1)) * c))
    )
    # Herbie: distribute-lft1-in
    yield rewrite((b * a) + a).to((b + Scalar.number(BigRat(1, 1))) * a)
    # Herbie: distribute-rgt1-in
    yield rewrite(a + (c * a)).to((c + Scalar.number(BigRat(1, 1))) * a)
    # Herbie: distribute-lft-neg-in
    yield rewrite(Scalar.number(BigRat(-1, 1)) * (a * b)).to((Scalar.number(BigRat(-1, 1)) * a) * b)
    # Herbie: distribute-rgt-neg-in
    yield rewrite(Scalar.number(BigRat(-1, 1)) * (a * b)).to(a * (Scalar.number(BigRat(-1, 1)) * b))
    # Herbie: distribute-lft-neg-out
    yield rewrite((Scalar.number(BigRat(-1, 1)) * a) * b).to(Scalar.number(BigRat(-1, 1)) * (a * b))
    # Herbie: distribute-rgt-neg-out
    yield rewrite(a * (Scalar.number(BigRat(-1, 1)) * b)).to(Scalar.number(BigRat(-1, 1)) * (a * b))
    # Herbie: distribute-neg-in
    yield rewrite(Scalar.number(BigRat(-1, 1)) * (a + b)).to(
        (Scalar.number(BigRat(-1, 1)) * a) + (Scalar.number(BigRat(-1, 1)) * b)
    )
    # Herbie: distribute-neg-out
    yield rewrite((Scalar.number(BigRat(-1, 1)) * a) + (Scalar.number(BigRat(-1, 1)) * b)).to(
        Scalar.number(BigRat(-1, 1)) * (a + b)
    )
    # Herbie: fp-cancel-sign-sub
    yield rewrite(a + (Scalar.number(BigRat(-1, 1)) * ((Scalar.number(BigRat(-1, 1)) * b) * c))).to(
        a + (b * c)
    )
    # Herbie: fp-cancel-sub-sign
    yield rewrite(a + ((Scalar.number(BigRat(-1, 1)) * b) * c)).to(
        a + (Scalar.number(BigRat(-1, 1)) * (b * c))
    )
    # Herbie: fp-cancel-sign-sub-inv
    yield rewrite(a + (b * c)).to(
        a + (Scalar.number(BigRat(-1, 1)) * ((Scalar.number(BigRat(-1, 1)) * b) * c))
    )
    # Herbie: fp-cancel-sub-sign-inv
    yield rewrite(a + (Scalar.number(BigRat(-1, 1)) * (b * c))).to(
        a + ((Scalar.number(BigRat(-1, 1)) * b) * c)
    )
    # Herbie: sub-negate
    yield rewrite(Scalar.number(BigRat(-1, 1)) * (b + (Scalar.number(BigRat(-1, 1)) * a))).to(
        a + (Scalar.number(BigRat(-1, 1)) * b)
    )
    # Herbie: sub-negate-rev
    yield rewrite(a + (Scalar.number(BigRat(-1, 1)) * b)).to(
        Scalar.number(BigRat(-1, 1)) * (b + (Scalar.number(BigRat(-1, 1)) * a))
    )
    # Herbie: add-flip
    yield rewrite(a + b).to(a + (Scalar.number(BigRat(-1, 1)) * (Scalar.number(BigRat(-1, 1)) * b)))
    # Herbie: add-flip-rev
    yield rewrite(a + (Scalar.number(BigRat(-1, 1)) * (Scalar.number(BigRat(-1, 1)) * b))).to(a + b)
    # Herbie: sqr-neg
    yield rewrite((Scalar.number(BigRat(-1, 1)) * x) * (Scalar.number(BigRat(-1, 1)) * x)).to(x * x)
    # Herbie: sqr-neg-rev
    yield rewrite(x * x).to((Scalar.number(BigRat(-1, 1)) * x) * (Scalar.number(BigRat(-1, 1)) * x))
    # Herbie: cube-neg
    yield rewrite(
        ((Scalar.number(BigRat(-1, 1)) * x) * (Scalar.number(BigRat(-1, 1)) * x))
        * (Scalar.number(BigRat(-1, 1)) * x)
    ).to(Scalar.number(BigRat(-1, 1)) * ((x * x) * x))
    # Herbie: cube-neg-rev
    yield rewrite(Scalar.number(BigRat(-1, 1)) * ((x * x) * x)).to(
        ((Scalar.number(BigRat(-1, 1)) * x) * (Scalar.number(BigRat(-1, 1)) * x))
        * (Scalar.number(BigRat(-1, 1)) * x)
    )
    # Herbie: cube-prod
    yield rewrite(((x * y) * (x * y)) * (x * y)).to(((x * x) * x) * ((y * y) * y))
    # Herbie: cube-mult
    yield rewrite((x * x) * x).to(x * (x * x))
    # Herbie: cube-prod-rev
    yield rewrite(((x * x) * x) * ((y * y) * y)).to(((x * y) * (x * y)) * (x * y))
    # Herbie: cube-unmult
    yield rewrite(x * (x * x)).to((x * x) * x)


@ruleset
def herbie_polynomials(a: Scalar, b: Scalar) -> Iterator[RewriteOrRule]:
    """Translated polynomials rules with definedness guards."""
    # Herbie: swap-sqr
    yield rewrite((a * b) * (a * b)).to((a * a) * (b * b))
    # Herbie: unswap-sqr
    yield rewrite((a * a) * (b * b)).to((a * b) * (a * b))
    # Herbie: difference-of-squares
    yield rewrite((a * a) + (Scalar.number(BigRat(-1, 1)) * (b * b))).to(
        (a + b) * (a + (Scalar.number(BigRat(-1, 1)) * b))
    )
    # Herbie: difference-of-sqr-1
    yield rewrite((a * a) + (Scalar.number(BigRat(-1, 1)) * Scalar.number(BigRat(1, 1)))).to(
        (a + Scalar.number(BigRat(1, 1)))
        * (a + (Scalar.number(BigRat(-1, 1)) * Scalar.number(BigRat(1, 1))))
    )
    # Herbie: difference-of-sqr--1
    yield rewrite((a * a) + Scalar.number(BigRat(-1, 1))).to(
        (a + Scalar.number(BigRat(1, 1)))
        * (a + (Scalar.number(BigRat(-1, 1)) * Scalar.number(BigRat(1, 1))))
    )
    # Herbie: sum-square-pow
    yield rewrite((a + b) * (a + b)).to(
        ((a * a) + (Scalar.number(BigRat(2, 1)) * (a * b))) + (b * b)
    )
    # Herbie: sub-square-pow
    yield rewrite(
        (a + (Scalar.number(BigRat(-1, 1)) * b)) * (a + (Scalar.number(BigRat(-1, 1)) * b))
    ).to(
        ((a * a) + (Scalar.number(BigRat(-1, 1)) * (Scalar.number(BigRat(2, 1)) * (a * b))))
        + (b * b)
    )
    # Herbie: sum-square-pow-rev
    yield rewrite(((a * a) + (Scalar.number(BigRat(2, 1)) * (a * b))) + (b * b)).to(
        (a + b) * (a + b)
    )
    # Herbie: sub-square-pow-rev
    yield rewrite(
        ((a * a) + (Scalar.number(BigRat(-1, 1)) * (Scalar.number(BigRat(2, 1)) * (a * b))))
        + (b * b)
    ).to((a + (Scalar.number(BigRat(-1, 1)) * b)) * (a + (Scalar.number(BigRat(-1, 1)) * b)))
    # Herbie: difference-of-sqr-1-rev
    yield rewrite(
        (a + Scalar.number(BigRat(1, 1)))
        * (a + (Scalar.number(BigRat(-1, 1)) * Scalar.number(BigRat(1, 1))))
    ).to((a * a) + (Scalar.number(BigRat(-1, 1)) * Scalar.number(BigRat(1, 1))))
    # Herbie: difference-of-sqr--1-rev
    yield rewrite(
        (a + Scalar.number(BigRat(1, 1)))
        * (a + (Scalar.number(BigRat(-1, 1)) * Scalar.number(BigRat(1, 1))))
    ).to((a * a) + Scalar.number(BigRat(-1, 1)))
    # Herbie: difference-of-squares-rev
    yield rewrite((a + b) * (a + (Scalar.number(BigRat(-1, 1)) * b))).to(
        (a * a) + (Scalar.number(BigRat(-1, 1)) * (b * b))
    )
    # Herbie: sum-cubes
    yield rewrite(((a * a) * a) + ((b * b) * b)).to(
        ((a * a) + ((b * b) + (Scalar.number(BigRat(-1, 1)) * (a * b)))) * (a + b)
    )
    # Herbie: difference-cubes
    yield rewrite(((a * a) * a) + (Scalar.number(BigRat(-1, 1)) * ((b * b) * b))).to(
        ((a * a) + ((b * b) + (a * b))) * (a + (Scalar.number(BigRat(-1, 1)) * b))
    )
    # Herbie: difference-cubes-rev
    yield rewrite(((a * a) + ((b * b) + (a * b))) * (a + (Scalar.number(BigRat(-1, 1)) * b))).to(
        ((a * a) * a) + (Scalar.number(BigRat(-1, 1)) * ((b * b) * b))
    )
    # Herbie: sum-cubes-rev
    yield rewrite(((a * a) + ((b * b) + (Scalar.number(BigRat(-1, 1)) * (a * b)))) * (a + b)).to(
        ((a * a) * a) + ((b * b) * b)
    )
