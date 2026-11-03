"""Optional structural checks at existing lowering boundaries.

These checks assume valid input nodes. They neither run transformations nor
check mathematical declarations or downstream compiler capabilities. A contract
is simply a function returning violations; an empty tuple means it is satisfied.
Stored geometry, spaces and labels are outside the expression traversal and must
be checked separately when exposed. Each shared expression is reported once.
"""

from __future__ import annotations

from collections.abc import Callable

from .nodes import AbstractExpression, DirectionalDerivative, Integral, Mesh
from .passes.gradients import allowed
from .substitution import expressions

Violation = tuple[AbstractExpression, str]


def _violations(
    expression: AbstractExpression, rule: Callable[[AbstractExpression], str | None]
) -> tuple[Violation, ...]:
    seen: set[AbstractExpression] = set()
    found: list[Violation] = []

    def visit(node: AbstractExpression) -> None:
        if node in seen:
            return
        seen.add(node)
        reason = rule(node)
        if reason is not None:
            found.append((node, reason))
        for child in expressions(node):
            visit(child)

    visit(expression)
    return tuple(found)


def _derivative(node: AbstractExpression) -> str | None:
    return "directional derivative remains" if isinstance(node, DirectionalDerivative) else None


def _reference_integral(node: AbstractExpression) -> str | None:
    if reason := _derivative(node):
        return reason
    if isinstance(node, Integral) and isinstance(node.integration_domain, Mesh):
        return "mesh integral remains"
    return None


def derivatives_expanded(expression: AbstractExpression) -> tuple[Violation, ...]:
    """Check that no directional derivatives remain in expression operands.

    Actions, spatial operators and mesh integrals are permitted. This does not
    certify preservation of an input's original spatial notation or domains.
    """
    return _violations(expression, _derivative)


def reference_integration(expression: AbstractExpression) -> tuple[Violation, ...]:
    """Check the derivative-free boundary after mesh integral expansion.

    No mesh integrals or directional derivatives may remain. Other core integral
    domains, actions, gradients and pull-backs are permitted. No quadrature rule
    or basis tabulation capability is required. This is a cumulative recipe
    boundary, not the postcondition of integral expansion on arbitrary input.
    """
    return _violations(expression, _reference_integral)


def core_normal_form(expression: AbstractExpression) -> tuple[Violation, ...]:
    """Diagnose violations of the interpreter's existing core normal form.

    Delegate allowed-node patterns to the existing predicate. Accepted terminal
    operations may need later compiler/library processing; that is not checked.
    """
    return _violations(
        expression,
        lambda node: (
            None if allowed(node) else f"{type(node).__name__} is outside core normal form"
        ),
    )
