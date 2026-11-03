from uflx_spec_interpreter._src.extensions.shape import lower_with_shape_derivatives

from .quadrature import apply_quadrature
from .simplification import simplify
from .tabulation import tabulate


def compile_form(f, degree):
    # The extension recipe finishes at the checked core boundary. The compiler
    # then owns quadrature, tabulation and simplification of array arithmetic.
    f = lower_with_shape_derivatives(f)
    return simplify(tabulate(apply_quadrature(f, degree)))
