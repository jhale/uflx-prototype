from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING

from ..algorithms.rewriting import rewrite
from ..algorithms.substitution import occurs_in_metadata
from ..algorithms.traversal import everywhere
from ..nodes import DirectionalDerivative
from ..nodes.arguments import argument_members
from ..rules.differentiation import deriv
from .applications import expand_applications

if TYPE_CHECKING:
    from ..nodes.expressions import AbstractExpression
    from ..rules.differentiation import DerivativeRule


def apply_derivatives(f: AbstractExpression) -> AbstractExpression:
    return expand_derivatives_with(f, None)


def expand_derivatives_with(
    f: AbstractExpression, rule: DerivativeRule | None
) -> AbstractExpression:
    """Expand directional derivatives with an optional rule for library nodes.

    The rule receives the node, a function to evaluate an operand at the
    differentiation point, and a function to differentiate an operand. Return
    a derivative expression or None to continue with the core rules.
    """
    differential = cache(deriv)

    def evaluate(e: AbstractExpression) -> AbstractExpression:
        match e:
            case DirectionalDerivative(g, x, a, h):
                variables = argument_members(x)
                if any(occurs_in_metadata(variable, g) for variable in variables):
                    # The bottom-up rewrite has resolved inner derivatives.
                    # Expose only this body's geometry before selecting a
                    # constant or chain rule; point and direction stay outside.
                    from .integrals import expand_integrals
                    from .spatial import lower_spatial

                    g = lower_spatial(
                        expand_integrals(expand_applications(g)),
                        lambda e: expand_derivatives_with(e, rule),
                    )
                    if any(occurs_in_metadata(variable, g) for variable in variables):
                        raise ValueError(
                            "differentiation has unresolved geometry or space dependence; "
                            "an explicit reference representation or derivative rule is required"
                        )
                return differential(g, x, a, h, rule)
            case _:
                return e

    # Applying a newly exposed action can reveal another derivative.
    seen: set[AbstractExpression] = set()
    while f not in seen:
        seen.add(f)
        f = expand_applications(rewrite(f, evaluate))
        if _derivatives_complete(f):
            return f
    raise ValueError("derivative rewriting made no progress", f)


def _derivatives_complete(e: AbstractExpression) -> bool:
    return everywhere(e, lambda n: not isinstance(n, DirectionalDerivative))
