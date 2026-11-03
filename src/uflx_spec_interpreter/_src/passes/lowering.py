"""The standard core lowering recipe and its checked output boundary."""

from ..contracts import core_normal_form
from ..nodes import AbstractExpression
from .actions import apply_actions
from .derivatives import apply_derivatives
from .gradients import lower as lower_spatial
from .integrals import expand_integrals


def lower(f: AbstractExpression) -> AbstractExpression:
    """Lower a core expression and verify its core normal form.

    Geometry exposed during lowering must satisfy the geometry precondition of
    the specification. Extension operations require their own recipe before
    entering this operation. Retained library field_terminals are not evaluated here.

    Raises:
        TypeError: The input is not an expression or a rewrite is unsupported.
            An unsupported output carries offending nodes and diagnostic reasons
            in the exception's second argument.
    """
    if not isinstance(f, AbstractExpression):
        raise TypeError("lowering requires an expression")
    out = lower_spatial(expand_integrals(apply_derivatives(apply_actions(f))))
    violations = core_normal_form(out)
    if violations:
        reasons = "; ".join(dict.fromkeys(reason for _, reason in violations))
        raise TypeError(f"unsupported core lowering: {reasons}", violations)
    return out
