"""Check the port against direct dictionary translations of the Haskell equations.

No Haskell compiler is required. The oracle deliberately copies whole maps and
recomputes their XOR, independently of the port's AVL/update implementation.
Both use the documented replacement hash primitives, not Haskell Hashable.
"""

import random

from uflx_spec_interpreter._src.algorithms._alpha_summary import (
    MASK,
    Hasher,
    Maps,
    digest,
    then_hash,
)


def test_integer_combiner_and_typed_payloads():
    rng = random.Random(190)
    for _ in range(1000):
        seed, value = rng.getrandbits(128), rng.getrandbits(128)
        expected = seed
        for word in (value, 0):
            expected = (expected ^ (word + 2654435769 + (expected << 6) + (expected >> 2))) & MASK
        assert then_hash(seed, value) == expected
    assert len({digest(None), digest(False), digest(True), digest(0), digest(1)}) == 5
    assert digest((1,)) != digest((1.0,))
    assert digest((0.0,)) != digest((-0.0,))
    assert then_hash(10, 20) != then_hash(20, 10)


def tagged(tag, *values):
    result = digest(tag)
    for value in values:
        result = then_hash(result, value)
    return result


def vm_hash(entries):
    result = 0
    for name, positions in entries.items():
        result ^= then_hash(digest(name), positions)
    return result


def assert_summary(actual, structure, variables, backend):
    assert actual.structure == structure
    assert actual.fingerprint == then_hash(digest(structure), vm_hash(variables))
    keys = {value: name for name, value in backend.names.items()}
    assert {keys[e.key]: e.position for e in backend.maps.entries(actual.variables)} == variables


def test_reference_equations_at_every_subterm():
    rng, h = random.Random(710), Hasher()

    def expression(depth):
        if depth == 0 or rng.randrange(4) == 0:
            name = rng.choice(("a", "b", "c", "free"))
            actual = h.variable(name)
            structure, variables = digest("SVar"), {name: digest("HerePL")}
        elif rng.randrange(2):
            name = rng.choice(("a", "b", "c"))
            body, child_structure, child_map = expression(depth - 1)
            variables = child_map.copy()
            positions = variables.pop(name, None)
            structure = tagged("SLam", child_structure, positions, child_structure)
            actual = h.lam(name, body)
        else:
            left, sl, ml = expression(depth - 1)
            right, sr, mr = expression(depth - 1)
            tag = then_hash(digest(sl), sr)
            left_bigger = len(ml) >= len(mr)
            big, small = (ml, mr) if left_bigger else (mr, ml)
            variables = big.copy()
            for name in sorted(small):
                variables[name] = tagged("JoinPL", tag, variables.get(name), small[name])
            structure = tagged("SApp", tag, left_bigger, sl, sr)
            actual = h.app(left, right)
        assert_summary(actual, structure, variables, h)
        return actual, structure, variables

    for _ in range(1000):
        expression(6)


def test_avl_persistence_balance_and_xor():
    maps, rng, tree, expected = Maps(), random.Random(9), None, {}
    snapshots = []

    def check(t):
        if t is None:
            return 0, 0, 0
        lh, ln, lx = check(t.left)
        rh, rn, rx = check(t.right)
        assert abs(lh - rh) <= 1
        assert t.height == 1 + max(lh, rh)
        assert t.size == 1 + ln + rn
        assert t.total == t.entry_hash ^ lx ^ rx
        return t.height, t.size, t.total

    for k in range(2000):
        key = rng.randrange(100)
        if rng.randrange(3):
            # Include zero: a zero hash is present, not a missing entry.
            position = rng.randrange(10)
            tree = maps.alter(tree, key, digest(key), lambda _: position)
            expected[key] = position
        else:
            tree, position = maps.pop(tree, key)
            assert position == expected.pop(key, None)
        check(tree)
        if k % 20 == 0:
            snapshots.append((tree, expected.copy()))
    for tree, expected in snapshots:
        entries = list(maps.entries(tree))
        assert [entry.key for entry in entries] == sorted(expected)
        assert {entry.key: entry.position for entry in entries} == expected
        assert (tree.total if tree else 0) == vm_hash(expected)


def test_larger_map_is_shared_and_work_tracks_smaller_map():
    h = Hasher()
    big = h.variable("0")
    for k in range(1, 1024):
        big = h.app(big, h.variable(str(k)))
    closed = h.lam("x", h.variable("x"))
    before = h.maps.updated_entries
    assert h.app(big, closed).variables is big.variables
    assert h.app(closed, big).variables is big.variables
    assert h.maps.updated_entries == before
    h.app(big, h.variable("0"))
    # One singleton creation and exactly one merged entry.
    assert h.maps.updated_entries == before + 2
