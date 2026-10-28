from uflx_spec_interpreter import (
    apply_actions,
    apply_derivatives,
    apply_gateaux_derivatives,
    apply_gradients,
    expand_integrals,
    expand_pull_backs,
)

from .evaluation import evaluate
from .quadrature import apply_quadrature
from .tabulation import coefficients, tabulate

__all__ = ["apply_quadrature", "coefficients", "compile_form", "evaluate", "tabulate"]


def compile_form(f, degree):
    # The passes of the core, then quadrature and tabulation, down to arithmetic on arrays.
    for apply in (
        apply_gateaux_derivatives,
        apply_actions,
        apply_derivatives,
        expand_integrals,
        expand_pull_backs,
        apply_gradients,
    ):
        f = apply(f)
    return tabulate(apply_quadrature(f, degree))
