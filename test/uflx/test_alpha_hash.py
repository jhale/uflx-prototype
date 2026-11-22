"""Alpha fingerprints preserve lexical identity and support cache reuse."""

import os
import random
import subprocess
import sys

import pytest

import uflx._src.algorithms._alpha_dependencies as sets
import uflx._src.algorithms.alpha as alpha
from uflx._src.algorithms.alpha import AlphaHasher, alpha_hash
from uflx._src.algorithms.binding import alpha_equivalent
from uflx._src.algorithms.substitution import rename_bound
from uflx.arithmetic import conjugate, num
from uflx.calculus import directional_derivative
from uflx.forms import form, function, rebind
from uflx.tensors import component_tensor, index, index_sum, tensor
from uflx_basix.spaces import space

from .test_alpha_equivalence import geometry


def assert_same(left, right):
    assert alpha_equivalent(left, right)
    assert alpha_hash(left) == alpha_hash(right)
    h = AlphaHasher()
    assert h.fingerprint(left) == h.fingerprint(right) == alpha_hash(left)


@pytest.mark.parametrize("kind", ["form", "tensor", "sum", "derivative"])
def test_all_binders(kind):
    V = space("hash-real", scalar_type="real")
    u, v, w = (function(V, s) for s in ("u", "v", "w"))
    i, j = index("i"), index("j")
    a = tensor((num(2), num(3)))
    match kind:
        case "form":
            left, right = form(u * w, (u,)), form(v * w, (v,))
        case "tensor":
            left, right = component_tensor(a[i], (i,)), component_tensor(a[j], (j,))
        case "sum":
            left, right = index_sum(a[i], (i,)), index_sum(a[j], (j,))
        case "derivative":
            left = directional_derivative(u * u, u, w, w)
            right = directional_derivative(v * v, v, w, w)
    assert_same(left, right)
    assert left is not right  # Ordinary interning still distinguishes names.


def test_shadowing_grouping_conjugation_and_free_identity():
    V = space("hash-complex", scalar_type="complex")
    u, v, w, p = (function(V, s) for s in ("u", "v", "w", "p"))
    left = form(form(u, (v,)) + form(u, (u,)), (u,))
    right = form(form(w, (p,)) + form(p, (p,)), (w,))
    assert_same(left, right)
    assert_same(
        form(u * conjugate(v), (u, conjugate(v))), form(w * conjugate(p), (w, conjugate(p)))
    )
    pairs = [
        (left, form(form(p, (p,)) + form(p, (p,)), (w,))),
        (form(u * v, (u,)), form(w * p, (w,))),
        (form(u * v, ((u, v),)), form(u * v, (u, v))),
        (form(u * v, (u, conjugate(v))), form(u * v, (u, v))),
        (form(u * v, (u, v)), form(u * v, (v, u))),
    ]
    for a, b in pairs:
        assert not alpha_equivalent(a, b)
        assert alpha_hash(a) != alpha_hash(b)


def test_outer_dependent_metadata():
    x, y, c = geometry()
    original = form(c, (x,))
    renamed = rename_bound(original, {x: y})
    assert_same(original, renamed)
    assert not alpha_equivalent(original.expression, renamed.expression)
    assert alpha_hash(original.expression) != alpha_hash(renamed.expression)


@pytest.mark.parametrize("labels", [(1, 1.0), (0.0, -0.0)])
def test_opaque_labels_preserve_numeric_types(labels):
    V = space("hash-labels", scalar_type="real")
    a, b = (function(V, ("label", label)) for label in labels)
    assert alpha_hash(a) != alpha_hash(b)
    assert_same(form(a, (a,)), form(b, (b,)))
    assert alpha_hash(function(V, ("label", form(a, (a,))))) != alpha_hash(
        function(V, ("label", form(b, (b,))))
    )


def test_deep_dag_and_clear(monkeypatch):
    V = space("hash-deep", scalar_type="real")
    u, v = function(V, "u"), function(V, "v")
    a, b = u, v
    for _ in range(2000):
        a, b = a + a, b + b
    a, b = form(a, (u,)), form(b, (v,))
    limit = sys.getrecursionlimit()
    assert_same(a, b)
    h = AlphaHasher()
    first = h.fingerprint(a)
    visits = len(h._memo)

    def unexpected_assembly(*args):
        pytest.fail("cached fingerprint recomputed a summary")

    with monkeypatch.context() as patch:
        patch.setattr(alpha, "_assemble", unexpected_assembly)
        assert h.fingerprint(a) == first
    assert len(h._memo) == visits
    h.clear()
    assert not h._memo and not h._info and not h._names
    assert h.fingerprint(a) == first
    assert sys.getrecursionlimit() == limit


def test_dependency_sets_match_python_sets_and_preserve_snapshots():
    rng = random.Random(83)
    snapshots = [(None, frozenset())]
    for name in range(80):
        snapshots.append((sets.singleton(name), frozenset((name,))))
    for _ in range(500):
        a, expected_a = rng.choice(snapshots)
        b, expected_b = rng.choice(snapshots)
        result = sets.union(a, b)
        removed = tuple(rng.randrange(80) for _ in range(3))
        result = sets.without(result, removed)
        expected = (expected_a | expected_b).difference(removed)
        snapshots.append((result, expected))
        assert sets.union(result, result) is result

    def keys(tree):
        if tree is not None:
            yield from keys(tree.left)
            yield tree.key
            yield from keys(tree.right)

    for tree, expected in snapshots:
        assert list(keys(tree)) == sorted(expected)
        assert {name for name in range(81) if sets.contains(tree, name)} == expected


def test_dependency_and_irrelevant_scope_scaling(monkeypatch):
    V = space("hash-scaling", scalar_type="real")
    u, v = function(V, "u"), function(V, "v")

    created = 0
    original_node = sets.node

    def counted_node(*args):
        nonlocal created
        created += 1
        return original_node(*args)

    monkeypatch.setattr(sets, "node", counted_node)

    def work(count, scoped):
        before = created
        body = num(1) if scoped else u
        for k in range(count):
            body = form(body, (u,)) + form(body, (v,)) if scoped else body + function(V, str(k))
        h = AlphaHasher()
        h.fingerprint(body)
        return len(h._memo), created - before

    for scoped, n in ((True, 12), (False, 512)):
        small, large = work(n, scoped), work(2 * n, scoped)
        assert large[0] < 3 * small[0]
        assert large[1] < 3 * small[1]


def test_cross_process_determinism():
    code = """
from uflx._src.algorithms.alpha import alpha_hash
from uflx.forms import form, function
from uflx_basix.spaces import space
v = function(space('hash', scalar_type='real'), 'v')
print(alpha_hash(form(v * v, (v,))))
"""
    outputs = set()
    for seed in ("1", "42", "random"):
        result = subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            text=True,
            capture_output=True,
            env=os.environ | {"PYTHONHASHSEED": seed},
        )
        outputs.add(result.stdout)
    assert len(outputs) == 1


def test_tuple_cache_keys_preserve_identity_and_hold_references():
    first, second = tuple([1]), tuple([1])
    key = alpha._key(first, ())
    assert key == alpha._key(first, ())
    assert hash(key) == hash(alpha._key(first, ()))
    assert key != alpha._key(second, ())
    assert key != alpha._key((1.0,), ())
    assert key[1].value is first


@pytest.mark.parametrize("kind", ["simple", "swap", "group", "conjugate", "shadow"])
def test_rebind_form_fingerprints(kind):
    V = space("hash-rebind", scalar_type="complex")
    u, v, w, p = (function(V, name) for name in ("u", "v", "w", "p"))
    match kind:
        case "simple":
            original, renames = form(u * u + w, (u,)), {u: v}
        case "swap":
            original, renames = form(u + num(2) * v, (u, v)), {u: v, v: u}
        case "group":
            original, renames = form(u + num(2) * v, ((u, v),)), {(u, v): (w, p)}
        case "conjugate":
            original = form(u * conjugate(v), (u, conjugate(v)))
            renames = {u: w, conjugate(v): conjugate(p)}
        case "shadow":
            original, renames = form(form(u, (w,)) + form(u, (u,)), (u,)), {u: v}
    assert_same(original, rebind(original, renames))


@pytest.mark.parametrize("constructor", [component_tensor, index_sum])
def test_rebind_index_fingerprints(constructor):
    i, j = index("i"), index("j")
    a = tensor((num(2), num(3)))
    original = constructor(a[i], (i,))
    assert_same(original, rebind(original, {i: j}))


def test_rebind_geometry_fingerprints():
    x, y, c = geometry()
    original = form(c, (x,))
    renamed = rebind(original, {x: y})
    assert_same(original, renamed)
    assert alpha_hash(original.expression) != alpha_hash(renamed.expression)


def test_rebind_changed_opaque_label_preserves_free_identity():
    V = space("hash-rebind-label", scalar_type="real")
    u, v = function(V, "u"), function(V, "v")
    c = function(V, ("library.coefficient", u))
    original = form(c, (u,))
    renamed = rebind(original, {u: v})
    assert renamed.expression.label == ("library.coefficient", v)
    assert not alpha_equivalent(original, renamed)
    assert alpha_hash(original) != alpha_hash(renamed)
