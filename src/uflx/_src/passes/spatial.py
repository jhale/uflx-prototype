"""Shared spatial fixed point, with derivative evaluation supplied by its caller."""

from collections.abc import Callable

from ..nodes import AbstractExpression
from .applications import expand_applications
from .gradients import apply_gradients
from .pullbacks import expand_pull_backs


def lower_spatial(
    f: AbstractExpression,
    evaluate_derivatives: Callable[[AbstractExpression], AbstractExpression],
) -> AbstractExpression:
    # apply_gradients and expand_pull_backs until neither rewrites, the stage of the core after
    # expand_integrals. A graph that the passes leave unchanged is the same object. Gradients on
    # the domains of the forms come first, so a gradient of the coordinates is the identity.
    seen = {f}
    while True:
        gradients = apply_gradients(f)
        pulled_back = expand_pull_backs(gradients)
        out = evaluate_derivatives(expand_applications(pulled_back))
        if gradients == f and pulled_back == gradients and out == pulled_back:
            return out
        if out in seen:
            raise ValueError("spatial lowering cycle", out)
        seen.add(out)
        f = out
