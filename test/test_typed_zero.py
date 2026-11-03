"""Zero transformations preserve the declared scalar type and other static attributes."""

import numpy as np
import pytest

from uflx_compiler.algebra import _normalizer
from uflx_compiler.evaluation import evaluate as evaluate_array
from uflx_compiler.simplification import instantiated, simplify
from uflx_oracle import evaluate
from uflx_spec_interpreter import (
    Integer,
    ScalarType,
    action,
    apply_actions,
    apply_derivatives,
    directional_derivative,
    function,
    index,
    num,
)
from uflx_spec_interpreter._src.literals import _zero
from uflx_spec_interpreter._src.nodes import Complex, Product, Sum, Zero
from uflx_toy.spaces import space


@pytest.mark.parametrize("scalar_type", (ScalarType.real, ScalarType.complex))
def test_zero_construction_and_reconstruction(scalar_type):
    k = index("typed-zero")
    V = space("slot")
    z = _zero((2,), ((k, 3),), (V,), scalar_type)
    assert z.scalar_type == scalar_type
    assert Zero(*z.construction_arguments) is z
    assert z is not _zero(
        (2,),
        ((k, 3),),
        (V,),
        ScalarType.real if scalar_type == ScalarType.complex else ScalarType.complex,
    )
    applied = apply_actions(action(z, 0, function(V, "v")))
    assert isinstance(applied, Zero)
    assert (
        applied.value_shape,
        applied.free_indices,
        applied.argument_spaces,
        applied.scalar_type,
    ) == ((2,), ((k, 3),), (), scalar_type)
    assert instantiated(applied, {k: 1}) is _zero((2,), (), (), scalar_type)
    assert np.iscomplexobj(evaluate(_zero((2,), (), (), scalar_type), {})([])) == (
        scalar_type == ScalarType.complex
    )


@pytest.mark.parametrize("bad", ["invalid", complex, None, 1])
def test_zero_rejects_invalid_scalar_type(bad):
    with pytest.raises(TypeError):
        _zero((), (), (), bad)


def test_cancellation_and_zero_products_preserve_complex_type():
    i = Complex(Integer(0), Integer(1))
    for e in (Sum(i, Product(num(-1), i)), Product(num(0), i)):
        out = _normalizer()(e)
        assert out is _zero((), (), (), ScalarType.complex)
    z = _zero((), (), (), ScalarType.complex)
    assert simplify(Product(z, num(2))) is z
    assert simplify(Sum(z, num(2))).scalar_type == ScalarType.complex
    normalized = _normalizer()(Sum(z, num(2)))
    assert normalized.scalar_type == ScalarType.complex
    assert _normalizer()(normalized) is normalized


@pytest.mark.parametrize("scalar_type", (ScalarType.real, ScalarType.complex))
def test_constant_derivative_preserves_output_type_not_direction_type(scalar_type):
    k = index("derivative-family")
    V = space("complex-variable")
    u = function(V, "u")
    h = function(V, "h")
    z = _zero((2,), ((k, 3),), (), scalar_type)
    out = apply_derivatives(directional_derivative(z, u, u, h))
    assert out is z


def test_array_evaluation_uses_complex_dtype_for_indexed_tensor_zero():
    k = index("array-zero")
    z = _zero((2,), ((k, 3),), (), ScalarType.complex)
    value = evaluate_array(z, {})
    assert value.shape == (3, 2)
    assert np.iscomplexobj(value)
    assert not np.any(value)
