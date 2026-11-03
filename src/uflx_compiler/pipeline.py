from uflx_spec_interpreter import apply_actions, apply_derivatives
from uflx_spec_interpreter._src.extensions.shape import _lower_with_shape_derivatives
from uflx_spec_interpreter._src.passes.algebra import _normalizer

from .quadrature import apply_quadrature
from .simplification import simplify
from .tabulation import tabulate


def compile_form(f, degree):
    # The extension recipe finishes at the checked core boundary. The compiler
    # then owns quadrature, tabulation and simplification of array arithmetic.
    normalize = _normalizer()
    f = normalize(apply_derivatives(apply_actions(f)))
    f = normalize(_lower_with_shape_derivatives(f))
    return simplify(tabulate(apply_quadrature(f, degree)))
