"""Not executed: Pyright must diagnose each marked use of a wrong type."""

from typing import assert_type

from uflx_spec_interpreter import (
    RD,
    Complex,
    Float,
    Integer,
    add,
    domain_map,
    gateaux_derivative,
    lower,
    num,
    tensor,
)
from uflx_spec_interpreter._src.contracts import (
    _core_normal_form,
    _derivatives_expanded,
    _reference_integration,
)
from uflx_spec_interpreter._src.nodes import AbstractExpression, Function, Mesh
from uflx_spec_interpreter._src.traversal import _bottom_up


def valid(f: Function, g: Function) -> None:
    assert_type(add(f, g), AbstractExpression)
    assert_type(lower(f), AbstractExpression)
    assert_type(Complex(Integer(1), Float(2.0)), Complex)
    assert_type(_bottom_up(f, lambda e: e), AbstractExpression)


def invalid(u: Function, v: Function, w: Function, z: Function, mesh: Mesh) -> None:
    gateaux_derivative(3, u, u, w)  # expect: reportArgumentType
    gateaux_derivative(u, num(1), u, w)  # expect: reportArgumentType
    gateaux_derivative(u, (u, num(1)), (u, v), (w, z))  # expect: reportArgumentType
    gateaux_derivative(u, u, u, num(1))  # expect: reportArgumentType
    gateaux_derivative(u, u, 1, w)  # expect: reportArgumentType
    _derivatives_expanded(3)  # expect: reportArgumentType
    _reference_integration(3)  # expect: reportArgumentType
    _core_normal_form(3)  # expect: reportArgumentType
    lower(1)  # expect: reportArgumentType
    domain_map(RD(0), tensor((num(0),)), mesh)  # expect: reportArgumentType
    tensor((1,))  # expect: reportArgumentType
    Complex(Integer(1) + Integer(2), Integer(0))  # expect: reportArgumentType
    _bottom_up(u, lambda e: e.domain)  # expect: reportArgumentType


def incomplete(node: Integer | Float) -> int | None:
    match node:  # expect: reportMatchNotExhaustive
        case Integer(n):
            return n
