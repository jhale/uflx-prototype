from .arithmetic import add, multiply
from .nodes import Float, ImaginaryUnit, Integer, Zero


def num(value):
    # A literal of the specification: an Integer, a Float, or a complex number as a composition.
    if isinstance(value, complex):
        return add(num(value.real), multiply(ImaginaryUnit(), num(value.imag)))
    return Integer(value) if isinstance(value, int) else Float(value)


def zero(value_shape, free_indices, argument_spaces):
    return Zero(value_shape, free_indices, argument_spaces)
