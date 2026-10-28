import gc
import os
import subprocess
import sys
import weakref
from dataclasses import dataclass, fields, replace

import pytest
from test_surface import Ms
from testutil import contains

from uflx_spec_interpreter import nodes
from uflx_spec_interpreter.domains import cell_domain
from uflx_spec_interpreter.geometry import density, inverse_jacobian, metric
from uflx_spec_interpreter.nodes import Base, Fn, Index, Indexed, Num, Space, Sum, Tensor
from uflx_spec_interpreter.substitution import map_fields

V = Space("V")


def test_equal_constructions_are_one_object():
    u = Fn(V, "u")
    assert Sum(Num(1), u) is Sum(Num(1), Fn(Space("V"), "u"))
    X = Tensor((Num(1), Num(2)))
    assert Index(("matmul", X, u, 0)) is Index(("matmul", Tensor((Num(1), Num(2))), u, 0))
    assert Space("V") is Space("V", dual=False)


def test_constants_differ_by_type_and_sign():
    values = [Num(1), Num(1.0), Num(True), Num(0.0), Num(-0.0), Num(1 + 0j)]
    assert len({id(v) for v in values}) == 6


def test_rebuilding_returns_the_same_object():
    e = Sum(Num(1), Fn(V, "u"))
    assert replace(e) is e
    assert map_fields(e, lambda x: x) is e


def test_unused_nodes_are_freed():
    ref = weakref.ref(Fn(V, ("unused", 12345)))
    gc.collect()
    assert ref() is None


def test_hash_is_the_same_in_every_process():
    code = (
        "from uflx_spec_interpreter.nodes import Fn, Indexed, Space, Sum, Tensor, Num\n"
        "u = Fn(Space('V'), ('label', 'with', 'strings'))\n"
        "print(hash(Sum(Indexed(Tensor((u, Num(2.5))), (1,)), u)))\n"
    )
    hashes = set()
    for seed in ("1", "2"):
        env = os.environ | {"PYTHONHASHSEED": seed}
        out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
        hashes.add(out.stdout.strip())
    assert len(hashes) == 1


def test_hash_does_not_depend_on_where_the_class_is():
    e = Sum(Num(1), Fn(V, "u"))
    digest = nodes.digest_of(Sum.identifier, [getattr(e, f.name) for f in fields(e)])
    module, qualname = Sum.__module__, Sum.__qualname__
    try:
        Sum.__module__, Sum.__qualname__ = "elsewhere", "Elsewhere"
        assert nodes.digest_of(Sum.identifier, [getattr(e, f.name) for f in fields(e)]) == digest
    finally:
        Sum.__module__, Sum.__qualname__ = module, qualname
    assert e.digest == digest


def test_a_class_with_no_identifier_raises():
    @dataclass(frozen=True, eq=False)
    class Anonymous(Base):
        value: int

    with pytest.raises(TypeError):
        Anonymous(1)


def test_density_and_inverse_jacobian_share_the_metric():
    m = cell_domain(Ms, 0)
    G = metric(m)
    assert contains(density(m), lambda n: n is G)
    assert contains(inverse_jacobian(m), lambda n: n is G)


def test_indexed_expressions_are_interned():
    T = Tensor((Num(1), Num(2)))
    assert Indexed(T, (Index("i"),)) is Indexed(Tensor((Num(1), Num(2))), (Index("i"),))
