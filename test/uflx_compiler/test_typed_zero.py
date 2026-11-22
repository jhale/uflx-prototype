"""Zero transformations preserve the declared scalar type and other static attributes."""

import numpy as np
import pytest

from uflx._src.literals import zero
from uflx._src.nodes import Complex, Product, Sum, Zero
from uflx.arithmetic import num
from uflx.forms import apply, function
from uflx.nodes import Integer, ScalarType
from uflx.passes import expand_applications
from uflx.tensors import index
from uflx_basix.spaces import space
from uflx_compiler.algebra import _normalizer
from uflx_compiler.evaluation import evaluate as evaluate_array
from uflx_compiler.simplification import instantiated, simplify
from uflx_oracle.interpreter import Interpreter


@pytest.mark.parametrize("scalar_type", (ScalarType.real, ScalarType.complex))
def test_zero_construction_and_reconstruction(scalar_type):
    k = index("typed-zero")
    V = space("slot")
    z = zero((2,), ((k, 3),), (V,), scalar_type)
    assert z.scalar_type == scalar_type
    assert Zero(*z.construction_arguments) is z
    assert z is not zero(
        (2,),
        ((k, 3),),
        (V,),
        ScalarType.real if scalar_type == ScalarType.complex else ScalarType.complex,
    )
    applied = expand_applications(apply(z, (function(V, "v"),)))
    assert isinstance(applied, Zero)
    assert (
        applied.value_shape,
        applied.free_indices,
        applied.argument_spaces,
        applied.scalar_type,
    ) == ((2,), ((k, 3),), (), scalar_type)
    assert instantiated(applied, {k: 1}) is zero((2,), (), (), scalar_type)
    assert np.iscomplexobj(Interpreter().evaluate(zero((2,), (), (), scalar_type))) == (
        scalar_type == ScalarType.complex
    )


def test_cancellation_and_zero_products_preserve_complex_type():
    i = Complex(Integer(0), Integer(1))
    for e in (Sum(i, Product(num(-1), i)), Product(num(0), i)):
        out = _normalizer()(e)
        assert out is zero((), (), (), ScalarType.complex)
    z = zero((), (), (), ScalarType.complex)
    assert simplify(Product(z, num(2))) is z
    assert simplify(Sum(z, num(2))).scalar_type == ScalarType.complex
    normalized = _normalizer()(Sum(z, num(2)))
    assert normalized.scalar_type == ScalarType.complex
    assert _normalizer()(normalized) is normalized


def test_array_evaluation_uses_complex_dtype_for_indexed_tensor_zero():
    k = index("array-zero")
    z = zero((2,), ((k, 3),), (), ScalarType.complex)
    value = evaluate_array(z, {})
    assert value.shape == (3, 2)
    assert np.iscomplexobj(value)
    assert not np.any(value)
