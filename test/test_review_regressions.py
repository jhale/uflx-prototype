"""Regressions for rewrite completion, extension boundaries and value substitution."""

from __future__ import annotations

import numpy as np
import pytest

from uflx_compiler.algebra import _normalizer
from uflx_compiler.evaluation import evaluate
from uflx_compiler.simplification import instantiated, simplify
from uflx_oracle import evaluate as oracle
from uflx_spec_interpreter import (
    Complex,
    Float,
    Integer,
    ScalarType,
    Zero,
    apply_derivatives,
    component_tensor,
    conjugate,
    directional_derivative,
    function,
    index,
    index_sum,
    lower,
    num,
    tensor,
)
from uflx_spec_interpreter._src.nodes import AbstractExpression, Base, IndexSum
from uflx_spec_interpreter._src.nodes.structure import expression_operands, operands
from uflx_toy.spaces import space


@pytest.mark.parametrize("rewrite", [lower, _normalizer(), simplify])
@pytest.mark.parametrize(
    "imaginary", [Integer(0), Float(0.0), Float(-0.0), Integer(3), Float(0.25)]
)
def test_complex_branch_arithmetic_survives_rewriting(rewrite, imaginary):
    e = conjugate(Complex(Integer(-1), imaginary)) ** Float(0.5)
    out = rewrite(e)
    assert out.scalar_type == ScalarType.complex
    assert np.isclose(evaluate(out, {}), evaluate(e, {}))
    assert np.isfinite(evaluate(out, {}))


def test_derivative_pass_does_not_implicitly_normalize_algebra():
    e = Integer(2) + Integer(3)
    assert apply_derivatives(e) is e
    assert _normalizer()(e) is Integer(5)


@pytest.mark.parametrize("bind", [index_sum, component_tensor])
def test_simplification_completes_a_child_before_binding_its_index(bind):
    i = index("lost-child-index")
    e = bind(Integer(1) + Zero((), ((i, 2),), (), ScalarType.real), (i,))
    out = simplify(e)
    assert out.free_indices == ()
    assert out.value_shape == e.value_shape
    assert np.allclose(evaluate(out, {}), evaluate(e, {}))
    assert simplify(out) is out


class ExternalOperation(AbstractExpression):
    identifier = "library.ExternalOperation"
    value_fields = ("operand",)
    operand: AbstractExpression
    value_shape = ()
    free_indices = ()
    argument_spaces = ()
    domain = None
    scalar_type = ScalarType.real


class LibraryTerminal(AbstractExpression):
    # Even a core-like identifier has no bearing on terminal status.
    identifier = "uflx.library.Terminal"
    is_terminal = True
    label: str
    value_shape = ()
    free_indices = ()
    argument_spaces = ()
    domain = None
    scalar_type = ScalarType.real


def test_unknown_library_operation_is_not_a_terminal():
    with pytest.raises(TypeError, match="ExternalOperation"):
        lower(ExternalOperation(Integer(2)))
    terminal = LibraryTerminal("data")
    assert lower(terminal) is terminal


def test_construction_dependencies_and_value_operands_are_distinct():
    V = space("dependencies")
    source = function(V, "source")
    f = function(V, ("derived", source))
    assert operands(f) == [V, source]
    assert expression_operands(f) == []
    c = Complex(Integer(1), Integer(2))
    assert operands(c) == [Integer(1), Integer(2)]
    assert expression_operands(c) == []


def test_at_point_derivative_does_not_substitute_into_inserted_direction():
    x = function(space("direct-derivative"), "x")
    e = directional_derivative(x * x, x, num(2), x)
    out = apply_derivatives(e)
    assert np.isclose(oracle(out, {x: 3})(()), 12)


def test_compiler_index_instantiation_avoids_capture():
    i, j = index("outer"), index("inner")
    a, b = tensor((num(2), num(3))), tensor((num(5), num(7)))
    e = index_sum(a[i] * b[j], (j,))
    out = instantiated(e, {i: j})
    assert isinstance(out, IndexSum)
    assert out.indices != (j,)
    assert out.free_indices == ((j, 2),)
    assert np.allclose(evaluate(out, {}), [24, 36])


def test_validation_does_not_dispatch_on_class_name():
    class Index(Base):
        identifier = "library.Index"
        value: int

    assert Index(3).value == 3
