from uflx import apply_derivatives, expand_applications, lower

from .algebra import _normalizer
from .quadrature import apply_quadrature
from .simplification import simplify
from .tabulation import tabulate


def compile_form(f, degree):
    # Core lowering includes geometry-aware differentiation. The compiler owns
    # quadrature, tabulation and simplification of array arithmetic.
    normalize = _normalizer()
    f = normalize(apply_derivatives(expand_applications(f)))
    f = normalize(lower(f))
    return simplify(tabulate(apply_quadrature(f, degree)))
