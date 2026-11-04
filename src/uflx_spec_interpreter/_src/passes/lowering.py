"""The standard core lowering recipe and its checked output boundary."""

from __future__ import annotations

from ..contracts import _core_normal_form
from ..nodes import AbstractExpression
from .actions import apply_actions
from .derivatives import apply_derivatives
from .gradients import apply_gradients
from .integrals import expand_integrals
from .pullbacks import expand_pull_backs

__all__ = ["_lower_spatial", "lower"]


def lower(f: AbstractExpression) -> AbstractExpression:
    """Lower a core expression and verify its core normal form.

    Geometry exposed during lowering must satisfy the geometry precondition of
    the specification. Extension operations require their own recipe before
    entering this operation. Retained library terminals are not evaluated here.

    Raises:
        TypeError: A rewrite is unsupported.
            An unsupported output carries offending nodes and diagnostic reasons
            in the exception's second argument.
    """
    out = _lower_spatial(expand_integrals(apply_derivatives(apply_actions(f))))
    violations = _core_normal_form(out)
    if violations:
        reasons = "; ".join(dict.fromkeys(reason for _, reason in violations))
        raise TypeError(f"unsupported core lowering: {reasons}", violations)
    return out


def _lower_spatial(f: AbstractExpression) -> AbstractExpression:
    # apply_gradients and expand_pull_backs until neither rewrites, the stage of the core after
    # expand_integrals. A graph that the passes leave unchanged is the same object. Gradients on
    # the domains of the forms come first, so a gradient of the coordinates is the identity.
    seen = {f}
    while True:
        gradients = apply_gradients(f)
        pulled_back = expand_pull_backs(gradients)
        out = apply_derivatives(apply_actions(pulled_back))
        if gradients == f and pulled_back == gradients and out == pulled_back:
            return out
        if out in seen:
            raise ValueError("spatial lowering cycle", out)
        seen.add(out)
        f = out
