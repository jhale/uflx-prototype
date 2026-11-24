# Copyright (C) 2026 Jack S. Hale
# Copyright (c) Microsoft Corporation.
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT
"""Port of AlphaHashOptimizedHash.hs's summary algorithm.

Algorithm reference:
    Krzysztof Maziarz, Tom Ellis, Alan Lawrence, Andrew Fitzgibbon, and
    Simon Peyton Jones. "Hashing Modulo Alpha-Equivalence." PLDI 2021.
    https://doi.org/10.1145/3453483.3454088
    https://arxiv.org/abs/2105.02856

Upstream implementation (Microsoft Corporation, MIT license retained below):
https://github.com/microsoft/hash-modulo-alpha/blob/main/src/AlphaHashOptimizedHash.hs

Var/Lam/App, the smaller-map fold, structure tags, left_bigger tie rule, optional
positions and XOR map aggregation follow the reference. An immutable AVL map
replaces Data.Map.Strict. The reference's two-round Boost-style integer combiner
uses unsigned 128-bit arithmetic here. BLAKE2 hashes non-integer payloads and
fixed tags; this is not a bit-for-bit port of Haskell's Hashable/Int primitives.

Summary operations are pure: callers supply integer lexical tokens, and all
occurrence maps are immutable.

UFLx adaptation is separate: constructor tags are closed constants, n-ary nodes
are left-folded applications, and ordered binders become nested lambdas. Exact
alpha comparison is still required after a hash match.
The deterministic combiner is not cryptographic and has no claimed adversarial
collision bound. Fingerprints are cache-bucket keys, never equality proofs.

MIT License

Copyright (c) Microsoft Corporation.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import lru_cache
from hashlib import blake2b


def payload_digest(value: object) -> int:
    """Hash a canonical internal payload without process-randomised Python hashes."""
    return int.from_bytes(blake2b(repr(value).encode(), digest_size=16).digest(), "big")


MASK = (1 << 128) - 1
NONE_HASH = payload_digest(("optional-position", "absent"))
FALSE_HASH = payload_digest(("bool", False))
TRUE_HASH = payload_digest(("bool", True))


@lru_cache(maxsize=1024)
def _string_hash(value: str) -> int:
    """Bounded cache for fixed tags and common names."""
    return payload_digest(value)


def digest(value: object) -> int:
    """Deterministic payload hash; already hashed integer words pass through."""
    if type(value) is int:
        return value & MASK
    if value is None:
        return NONE_HASH
    if type(value) is bool:
        return TRUE_HASH if value else FALSE_HASH
    if type(value) is str:
        return _string_hash(value)
    # Do not cache tuples by Python equality: (1,) and (1.0,) are distinct
    # construction payloads despite comparing equal in Python.
    return payload_digest(value)


def then_hash(seed: int, value: object) -> int:
    """Reference hashCombine's two rounds, with explicit 128-bit wrapping."""
    seed &= MASK
    seed = (seed ^ (digest(value) + 2654435769 + (seed << 6) + (seed >> 2))) & MASK
    return (seed ^ (2654435769 + (seed << 6) + (seed >> 2))) & MASK


def combine(tag: str, *values: object) -> int:
    result = digest(tag)
    for value in values:
        result = then_hash(result, value)
    return result


@dataclass(frozen=True, eq=False, slots=True)
class Entry:
    key: int
    position: int
    entry_hash: int
    left: Entry | None
    right: Entry | None
    height: int
    size: int
    total: int


def height(tree: Entry | None) -> int:
    return tree.height if tree else 0


def size(tree: Entry | None) -> int:
    return tree.size if tree else 0


def total(tree: Entry | None) -> int:
    return tree.total if tree else 0


def node(
    key: int, position: int, entry_hash: int, left: Entry | None, right: Entry | None
) -> Entry:
    return Entry(
        key,
        position,
        entry_hash,
        left,
        right,
        1 + max(height(left), height(right)),
        1 + size(left) + size(right),
        entry_hash ^ total(left) ^ total(right),
    )


def rebuild(tree: Entry, left: Entry | None, right: Entry | None) -> Entry:
    return node(tree.key, tree.position, tree.entry_hash, left, right)


def rotate_left(tree: Entry) -> Entry:
    right = tree.right
    assert right is not None
    return rebuild(right, rebuild(tree, tree.left, right.left), right.right)


def rotate_right(tree: Entry) -> Entry:
    left = tree.left
    assert left is not None
    return rebuild(left, left.left, rebuild(tree, left.right, tree.right))


def balanced(tree: Entry) -> Entry:
    if height(tree.left) > height(tree.right) + 1:
        assert tree.left is not None
        if height(tree.left.right) > height(tree.left.left):
            tree = rebuild(tree, rotate_left(tree.left), tree.right)
        return rotate_right(tree)
    if height(tree.right) > height(tree.left) + 1:
        assert tree.right is not None
        if height(tree.right.left) > height(tree.right.right):
            tree = rebuild(tree, tree.left, rotate_right(tree.right))
        return rotate_left(tree)
    return tree


def alter(
    tree: Entry | None,
    key: int,
    update: Callable[[int | None], int],
) -> Entry:
    if tree is None or tree.key == key:
        position = update(tree.position if tree else None)
        return node(
            key,
            position,
            then_hash(digest(key), position),
            tree.left if tree else None,
            tree.right if tree else None,
        )
    if key < tree.key:
        return balanced(rebuild(tree, alter(tree.left, key, update), tree.right))
    return balanced(rebuild(tree, tree.left, alter(tree.right, key, update)))


def pop(tree: Entry | None, key: int) -> tuple[Entry | None, int | None]:
    if tree is None:
        return None, None
    if key < tree.key:
        left, position = pop(tree.left, key)
        return (tree if position is None else balanced(rebuild(tree, left, tree.right))), position
    if key > tree.key:
        right, position = pop(tree.right, key)
        return (tree if position is None else balanced(rebuild(tree, tree.left, right))), position
    if tree.left is None:
        return tree.right, tree.position
    if tree.right is None:
        return tree.left, tree.position
    successor = tree.right
    while successor.left is not None:
        successor = successor.left
    right, _ = pop(tree.right, successor.key)
    return balanced(rebuild(successor, tree.left, right)), tree.position


def entries(tree: Entry | None) -> Iterator[Entry]:
    pending: list[Entry] = []
    while tree is not None or pending:
        while tree is not None:
            pending.append(tree)
            tree = tree.left
        tree = pending.pop()
        yield tree
        tree = tree.right


@dataclass(frozen=True, eq=False, slots=True)
class Summary:
    structure: int
    variables: Entry | None

    @property
    def fingerprint(self) -> int:
        # Haskell: hash structure `thenHash` hashVM positionMap.
        return then_hash(digest(self.structure), total(self.variables))


def variable(token: int) -> Summary:
    """Summarise one occurrence of an explicitly supplied lexical token."""
    position = digest("HerePL")
    return Summary(
        digest("SVar"), node(token, position, then_hash(digest(token), position), None, None)
    )


def bind(tokens: tuple[int, ...], body: Summary) -> Summary:
    """Bind ordered, distinct tokens without changing the input summary."""
    if len(set(tokens)) != len(tokens):
        raise ValueError("duplicate binding slots")
    for token in reversed(tokens):
        variables, positions = pop(body.variables, token)
        body = Summary(combine("SLam", body.structure, positions, body.structure), variables)
    return body


def application(left: Summary, right: Summary) -> Summary:
    """Combine two summaries, updating only entries of the smaller occurrence map."""
    tag = then_hash(digest(left.structure), right.structure)
    left_bigger = size(left.variables) >= size(right.variables)
    big, small = (left, right) if left_bigger else (right, left)
    variables = big.variables
    for entry in entries(small.variables):
        variables = alter(
            variables,
            entry.key,
            lambda previous: combine("JoinPL", tag, previous, entry.position),
        )
    return Summary(combine("SApp", tag, left_bigger, left.structure, right.structure), variables)


def constructor(tag: object, children: tuple[Summary, ...] = ()) -> Summary:
    """Encode a tagged UFLx constructor as a closed constant and applications."""
    result = Summary(combine("Constant", tag, len(children)), None)
    for child in children:
        result = application(result, child)
    return result
