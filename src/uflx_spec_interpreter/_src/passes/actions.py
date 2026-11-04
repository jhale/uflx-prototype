from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from ..nodes import Action, Form
from ..rewriting import _rewrite
from ..rules.binding import _bind
from ..rules.linearity import _action_linearity
from ..rules.structural import _action_retained, _action_structure, _structural
from ..static import _arity
from ..traversal import _everywhere, _memoized
from ..tuples import _members

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
        case Form(e, _) if p >= _arity(e):
            return _bind(f, p - _arity(e), g)
        case _:
            pass
    out = _action_structure(f, p, g, act)
    if out is None:
        out = _action_linearity(f, p, g, act)
    if out is None:
        out = _structural(f, lambda b: act(b, p, g), _members(g))
    if out is None:
        raise TypeError(f"no action rule for {type(f).__name__}")
    return out


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
