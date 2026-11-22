"""Not executed: Pyright must diagnose each marked use of a wrong type."""

from typing import Literal, assert_type

from uflx._src.passes.contracts import (
    core_normal_form,
    derivatives_expanded,
    reference_integration,
)
from uflx.algorithms import bottom_up
from uflx.arithmetic import add, num
from uflx.calculus import directional_derivative
from uflx.forms import form
from uflx.geometry import domain_map, side_map
from uflx.nodes import (
    AbstractDomain,
    AbstractExpression,
    AbstractTopology,
    Boolean,
    BroadcastTo,
    CellMap,
    Complex,
    DirectionalDerivative,
    DomainMap,
    EuclideanSpace,
    Float,
    Function,
    Gradient,
    Integer,
    Mesh,
    SideMap,
    Zero,
)
from uflx.passes import lower
from uflx.tensors import tensor


def valid(f: Function, g: Function) -> None:
    assert_type(Boolean(True).scalar_type, Literal["boolean"])
    assert_type(EuclideanSpace(2).dimension, int)
    assert_type(Gradient(f, EuclideanSpace(2)).domain, AbstractDomain)
    assert_type(Zero((), (), (), "real").scalar_type, Literal["real", "complex"])
    match Gradient(f, EuclideanSpace(2)):
        case Gradient(expression, domain):
            assert_type(expression, AbstractExpression)
            assert_type(domain, AbstractDomain)
    assert_type(directional_derivative(f, f, g), DirectionalDerivative)
    assert_type(directional_derivative(f, f, direction=g), DirectionalDerivative)
    assert_type(add(f, g), AbstractExpression)
    assert_type(lower(f), AbstractExpression)
    assert_type(Complex(Integer(1), Float(2.0)), Complex)
    assert_type(bottom_up(f, lambda e: e), AbstractExpression)


def invalid(u: Function, v: Function, w: Function, z: Function, mesh: Mesh) -> None:
    directional_derivative(u, u, 1)  # expect: reportArgumentType
    directional_derivative(u, u, point=w)  # expect: reportCallIssue
    directional_derivative(3, u, u, w)  # expect: reportArgumentType
    directional_derivative(u, num(1), u, w)  # expect: reportArgumentType
    directional_derivative(u, (u, num(1)), (u, v), (w, z))  # expect: reportArgumentType
    directional_derivative(u, u, 1, w)  # expect: reportArgumentType
    form(directional_derivative(u, u, u, num(1)), (num(1),))  # expect: reportArgumentType
    derivatives_expanded(3)  # expect: reportArgumentType
    reference_integration(3)  # expect: reportArgumentType
    core_normal_form(3)  # expect: reportArgumentType
    lower(1)  # expect: reportArgumentType
    EuclideanSpace("2")  # expect: reportArgumentType
    Gradient(u, 2)  # expect: reportArgumentType
    BroadcastTo(u, ("2",))  # expect: reportArgumentType
    Zero((), (), (), Boolean(True).scalar_type)  # expect: reportArgumentType
    Zero((), (), (), "integer")  # expect: reportArgumentType
    domain_map(EuclideanSpace(0), tensor((num(0),)), mesh)  # expect: reportArgumentType
    tensor((1,))  # expect: reportArgumentType
    Complex(Integer(1) + Integer(2), Integer(0))  # expect: reportArgumentType
    bottom_up(u, lambda e: e.domain)  # expect: reportArgumentType


def incomplete(node: Integer | Float) -> int | None:
    match node:  # expect: reportMatchNotExhaustive
        case Integer(n):
            return n


def cell_selection(topology: AbstractTopology, mesh: Mesh) -> None:
    assert_type(topology[0], CellMap)
    assert_type(mesh[0], CellMap)
    topology["cell"]  # expect: reportArgumentType
    mesh[:]  # expect: reportArgumentType


def side_selection(source: AbstractTopology, cell: CellMap, inclusion: DomainMap) -> None:
    assert_type(side_map(source, (cell, inclusion)), SideMap)
    side_map(source, (0, inclusion))  # expect: reportArgumentType
