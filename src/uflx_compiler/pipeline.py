from uflx.passes import apply_derivatives, expand_applications, lower
from uflx_basix.quadrature import apply_quadrature

from .algebra import _normalizer
from .simplification import simplify
from .tabulation import tabulate


def compile_form(f, degree):
    # Core lowering includes geometry-aware differentiation. Apply Basix quadrature,
    # then tabulate and simplify the resulting array arithmetic.
    normalize = _normalizer()
    f = normalize(apply_derivatives(expand_applications(f)))
    f = normalize(lower(f))
    return simplify(tabulate(apply_quadrature(f, degree)))
