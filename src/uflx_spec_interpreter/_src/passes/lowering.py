"""The standard core lowering recipe and its checked output boundary."""

from __future__ import annotations

from ..nodes import AbstractExpression
from .actions import apply_actions
from .arithmetic import expand_arithmetic
from .contracts import core_normal_form
from .derivatives import apply_derivatives
from .gradients import apply_gradients
from .integrals import expand_integrals
from .pullbacks import expand_pull_backs


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
    out = apply_actions(f)
    out = apply_derivatives(out)
    out = expand_integrals(out)
    out = lower_spatial(out)
    out = expand_arithmetic(out)
    violations = core_normal_form(out)
    if violations:
        reasons = "; ".join(dict.fromkeys(reason for _, reason in violations))
        raise TypeError(f"unsupported core lowering: {reasons}", violations)
    return out


def lower_spatial(f: AbstractExpression) -> AbstractExpression:
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
