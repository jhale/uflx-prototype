from uflx_spec_interpreter import (
    apply_actions,
    apply_derivatives,
    apply_gateaux_derivatives,
    apply_gradients,
    expand_integrals,
    expand_pull_backs,
)
from uflx_spec_interpreter._draft import apply_shape_derivatives

from .quadrature import apply_quadrature
from .simplification import simplify
from .tabulation import tabulate


def compile_form(f, degree):
    # The passes of the core, then quadrature and tabulation, down to arithmetic on arrays.
    for apply in (
        apply_gateaux_derivatives,
        apply_actions,
        apply_derivatives,
        expand_integrals,
        apply_gradients,
        expand_pull_backs,
        apply_shape_derivatives,
        apply_gradients,
        expand_pull_backs,
    ):
        f = apply(f)
    return simplify(tabulate(apply_quadrature(f, degree)))
