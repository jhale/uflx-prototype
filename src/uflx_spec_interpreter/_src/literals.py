from .nodes import Complex, Float, Integer, Zero


def num(value):
    # A convenience for the interpreter examples; expression factories never convert numbers.
    if type(value) is complex:
        return Complex(Float(value.real), Float(value.imag))
    return Integer(value) if type(value) is int else Float(value)


def zero(value_shape, free_indices, argument_spaces, scalar_type):
    return Zero(value_shape, free_indices, argument_spaces, scalar_type)
