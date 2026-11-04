from __future__ import annotations

import gc
import os
import subprocess
import sys
import weakref
from typing import ClassVar

import pytest
from test_surface import M
from testutil import contains

from uflx_spec_interpreter import density, inverse_jacobian
from uflx_spec_interpreter._src.algorithms.traversal import map_fields, rebuild
from uflx_spec_interpreter._src.geometry import jacobian
from uflx_spec_interpreter._src.meshes import cell_map
from uflx_spec_interpreter._src.nodes import (
    Base,
    Complex,
    Float,
    Function,
    Index,
    Indexed,
    Integer,
    Sum,
    Tensor,
)
from uflx_spec_interpreter._src.nodes.base import _digest_of
from uflx_toy.spaces import space

V = space("V")


def test_equal_constructions_are_one_object():
    u = Function(V, "u")
    assert Sum(Integer(1), u) is Sum(Integer(1), Function(space("V"), "u"))
    X = Tensor((Integer(1), Integer(2)))
    assert Index(("matmul", X, u, 0)) is Index(("matmul", Tensor((Integer(1), Integer(2))), u, 0))
    assert space("V") is space("V")


def test_literals_take_exact_types():
    values = [Integer(1), Float(1.0), Float(0.0), Float(-0.0), Complex(Integer(0), Integer(1))]
    assert len({id(v) for v in values}) == 5
    assert Complex(Integer(0), Integer(1)) is Complex(Integer(0), Integer(1))
    with pytest.raises(TypeError):
        Integer(True)
    with pytest.raises(TypeError):
        Float(1)
    with pytest.raises(ValueError):
        Float(float("inf"))


def test_rebuilding_returns_the_same_object():
    e = Sum(Integer(1), Function(V, "u"))
    assert rebuild(e) is e
    assert map_fields(e, lambda x: x) is e


def test_unused_nodes_are_freed():
    ref = weakref.ref(Function(V, ("unused", 12345)))
    gc.collect()
    assert ref() is None


def test_hash_is_the_same_in_every_process():
    code = (
        "from uflx_toy.spaces import space\n"
        "from uflx_spec_interpreter._src.nodes import "
        "Float, Function, Indexed, Integer, Sum, Tensor\n"
        "u = Function(space('V'), ('label', 'with', 'strings'))\n"
        "print(hash(Sum(Indexed(Tensor((u, Float(2.5))), (1,)), u)))\n"
    )
    hashes = set()
    for seed in ("1", "2"):
        env = os.environ | {"PYTHONHASHSEED": seed}
        out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
        assert out.returncode == 0, out.stderr
        hashes.add(out.stdout.strip())
    assert len(hashes) == 1 and hashes != {""}


def test_hash_does_not_depend_on_where_the_class_is():
    e = Sum(Integer(1), Function(V, "u"))
    digest = _digest_of(Sum.identifier, e.construction_arguments)
    module, qualname = Sum.__module__, Sum.__qualname__
    try:
        Sum.__module__, Sum.__qualname__ = "elsewhere", "Elsewhere"
        assert _digest_of(Sum.identifier, e.construction_arguments) == digest
    finally:
        Sum.__module__, Sum.__qualname__ = module, qualname
    assert e.digest == digest


def test_a_class_with_no_identifier_raises():
    class Anonymous(Base):
        value: int

    with pytest.raises(TypeError):
        Anonymous(1)


def test_density_and_inverse_jacobian_share_the_jacobian():
    m = cell_map(M, 0)
    J = jacobian(m)
    assert contains(density(m), lambda n: n is J)
    assert contains(inverse_jacobian(m), lambda n: n is J)


def test_indexed_expressions_are_interned():
    T = Tensor((Integer(1), Integer(2)))
    assert Indexed(T, (Index("i"),)) is Indexed(Tensor((Integer(1), Integer(2))), (Index("i"),))


def test_labels_that_read_alike_are_distinct():
    # The text of the encoding delimits strings, tuples and numbers, so none run together.
    labels = [("a,b",), ("a", "b"), "1", (1,), ((1,),), "(1)", "None", "#0"]
    functions = [Function(V, label) for label in labels]
    assert len({id(f) for f in functions}) == len(labels)


def test_cache_hits_do_not_initialize_or_validate_again():
    class Counted(Base):
        identifier = "test.Counted"
        value: int
        validations: ClassVar[int] = 0

        def _validate(self):
            type(self).validations += 1
            if self.value < 0:
                raise ValueError("negative value")

    node = Counted(2)
    assert Counted(2) is node
    assert Counted.validations == 1
    for _ in range(2):
        with pytest.raises(ValueError, match="negative value"):
            Counted(-1)
    assert Counted.validations == 3


def test_private_dataclass_storage_is_frozen():
    from dataclasses import FrozenInstanceError

    from uflx_spec_interpreter import RD

    domain = RD(3)
    with pytest.raises(FrozenInstanceError):
        domain._dimension = 4
    with pytest.raises(FrozenInstanceError):
        del domain._dimension
    assert domain.dimension == 3
    assert domain.construction_arguments == (3,)
    match domain:
        case RD(dimension):
            assert dimension == 3
        case _:
            pytest.fail("dataclass positional pattern did not match")
