"""Not executed: Pyright must diagnose each marked use of a wrong type."""

from typing import Literal, assert_type

from uflx_spec_interpreter import (
    RD,
    Complex,
    Float,
    Integer,
    Zero,
    add,
    domain_map,
    gateaux_derivative,
    lower,
    num,
    tensor,
)
from uflx_spec_interpreter._src.algorithms.traversal import bottom_up
from uflx_spec_interpreter._src.nodes import (
    AbstractDomain,
    AbstractExpression,
    BroadcastTo,
    Function,
    Gradient,
    Mesh,
)
from uflx_spec_interpreter._src.passes.contracts import (
    core_normal_form,
    derivatives_expanded,
    reference_integration,
)


def valid(f: Function, g: Function) -> None:
    assert_type(RD(2).dimension, int)
    assert_type(Gradient(f, RD(2)).domain, AbstractDomain)
    assert_type(Zero((), (), (), "real").scalar_type, Literal["real", "complex"])
    match Gradient(f, RD(2)):
        case Gradient(expression, domain):
            assert_type(expression, AbstractExpression)
            assert_type(domain, AbstractDomain)
    assert_type(add(f, g), AbstractExpression)
    assert_type(lower(f), AbstractExpression)
    assert_type(Complex(Integer(1), Float(2.0)), Complex)
    assert_type(bottom_up(f, lambda e: e), AbstractExpression)


def invalid(u: Function, v: Function, w: Function, z: Function, mesh: Mesh) -> None:
    gateaux_derivative(3, u, u, w)  # expect: reportArgumentType
    gateaux_derivative(u, num(1), u, w)  # expect: reportArgumentType
    gateaux_derivative(u, (u, num(1)), (u, v), (w, z))  # expect: reportArgumentType
    gateaux_derivative(u, u, u, num(1))  # expect: reportArgumentType
    gateaux_derivative(u, u, 1, w)  # expect: reportArgumentType
    derivatives_expanded(3)  # expect: reportArgumentType
    reference_integration(3)  # expect: reportArgumentType
    core_normal_form(3)  # expect: reportArgumentType
    lower(1)  # expect: reportArgumentType
    RD("2")  # expect: reportArgumentType
    Gradient(u, 2)  # expect: reportArgumentType
    BroadcastTo(u, ("2",))  # expect: reportArgumentType
    Zero((), (), (), "integer")  # expect: reportArgumentType
    domain_map(RD(0), tensor((num(0),)), mesh)  # expect: reportArgumentType
    tensor((1,))  # expect: reportArgumentType
    Complex(Integer(1) + Integer(2), Integer(0))  # expect: reportArgumentType
    bottom_up(u, lambda e: e.domain)  # expect: reportArgumentType


def incomplete(node: Integer | Float) -> int | None:
    match node:  # expect: reportMatchNotExhaustive
        case Integer(n):
            return n
