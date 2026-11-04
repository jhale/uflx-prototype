from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from ..arithmetic import multiply
from ..nodes import Action, Conjugate, Form, Product, Zero
from ..rewriting import _rewrite
from ..rules.binding import _bind
from ..rules.linearity import _linear_fields
from ..rules.structural import _action_retained, _commute, _structural_fields
from ..static import _free_indices
from ..traversal import _everywhere, _memoized
from ..tuples import _argument_members, _map_argument

if TYPE_CHECKING:
    from ..nodes.base import AbstractExpression


# Shared implementation helpers; the package public API is declared in __init__.py.
__all__ = ["_actions_complete", "apply_actions"]


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
            return _bind(f, p - len(e.argument_spaces), g)
        case Conjugate(expression):  # L05.
            return Conjugate(act(expression, p, _map_argument(Conjugate, g)))
        case Product(first, second):  # ST02.
            first_arity = len(first.argument_spaces)
            if p < first_arity:
                return multiply(act(first, p, g), second)
            return multiply(first, act(second, p - first_arity, g))
        case Zero(shape, _, arguments):  # L07.
            remaining = arguments[:p] + arguments[p + 1 :]
            return Zero(shape, _free_indices(f, *_argument_members(g)), remaining, f.scalar_type)
        case _ if _action_retained(f):  # ST03-ST05.
            return Action(f, p, g)
        case _:
            fields = _linear_fields(f) or _structural_fields(f)
            if fields is None:
                raise TypeError(f"no action rule for {type(f).__name__}")
            return _commute(
                f,
                lambda operand: act(operand, p, g),
                fields,
                _argument_members(g),
                lambda operand: not operand.argument_spaces,
            )


def apply_actions(f: AbstractExpression) -> AbstractExpression:
    # Inner actions first, then the rule for the form of each action.
    act = _memoized(_rule)

    def apply(e: AbstractExpression) -> AbstractExpression:
        match e:
            case Action(form, p, g):
                return act(form, p, g)
            case _:
                return e

    return _rewrite(f, apply)


def _actions_complete(e: AbstractExpression) -> bool:
    return _everywhere(e, lambda n: not isinstance(n, Action) or _action_retained(n.form))
