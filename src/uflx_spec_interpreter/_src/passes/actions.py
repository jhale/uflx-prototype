from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from ..algorithms.rewriting import rewrite
from ..algorithms.traversal import everywhere, memoized
from ..arithmetic import multiply
from ..nodes import Action, Conjugate, Form, Product, Zero
from ..nodes.arguments import argument_members, map_argument
from ..nodes.static import free_indices
from ..rules.binding import bind
from ..rules.linearity import linear_fields
from ..rules.structural import action_retained, commute, structural_fields

if TYPE_CHECKING:
    from ..nodes.expressions import AbstractExpression


def _rule(
    act: Callable[
        [AbstractExpression, int, AbstractExpression | tuple[AbstractExpression, ...]],
        AbstractExpression,
    ],
    f: AbstractExpression,
    p: int,
    g: AbstractExpression | tuple[AbstractExpression, ...],
) -> AbstractExpression:
    # The rule table of apply_actions, keyed on the form f.
    match f:
        case Form(e, _) if p >= len(e.argument_spaces):
            return bind(f, p - len(e.argument_spaces), g)
        case Conjugate(expression):  # L05.
            return Conjugate(act(expression, p, map_argument(Conjugate, g)))
        case Product(first, second):  # ST02.
            first_arity = len(first.argument_spaces)
            if p < first_arity:
                return multiply(act(first, p, g), second)
            return multiply(first, act(second, p - first_arity, g))
        case Zero(shape, _, arguments):  # L07.
            remaining = arguments[:p] + arguments[p + 1 :]
            return Zero(shape, free_indices(f, *argument_members(g)), remaining, f.scalar_type)
        case _ if action_retained(f):  # ST03-ST05.
            return Action(f, p, g)
        case _:
            fields = linear_fields(f) or structural_fields(f)
            if fields is None:
                raise TypeError(f"no action rule for {type(f).__name__}")
            return commute(
                f,
                lambda operand: act(operand, p, g),
                fields,
                argument_members(g),
                lambda operand: not operand.argument_spaces,
            )


def apply_actions(f: AbstractExpression) -> AbstractExpression:
    # Inner actions first, then the rule for the form of each action.
    act = memoized(_rule)

    def apply(e: AbstractExpression) -> AbstractExpression:
        match e:
            case Action(form, p, g):
                return act(form, p, g)
            case _:
                return e

    return rewrite(f, apply)


def actions_complete(e: AbstractExpression) -> bool:
    return everywhere(e, lambda n: not isinstance(n, Action) or action_retained(n.form))
