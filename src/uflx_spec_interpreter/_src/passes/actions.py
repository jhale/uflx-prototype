from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar, cast

from ..arithmetic import multiply
from ..nodes import Action, Conjugate, Form, Function, Gradient, Product, PullBack, Zero
from ..nodes.forms import _entry_function
from ..rewriting import _promote, _rewrite
from ..static import _arity, _field_terminal, _free_indices
from ..substitution import _freshen, _substitute
from ..traversal import _everywhere, _memoized
from ..tuples import _members
from .commuting import _commute

if TYPE_CHECKING:
    from ..nodes.base import AbstractExpression


_T = TypeVar("_T")


# Shared implementation helpers; the package public API is declared in __init__.py.
__all__ = ["_actions_complete", "_retained", "apply_actions"]


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
        case Conjugate(e):
            from ..tuples import _each

            return Conjugate(act(e, p, _each(Conjugate, g)))
        case Form(e, _) if p >= _arity(e):
            return _bind(f, p - _arity(e), g)
        case Product(a, b) if p < _arity(a):
            return multiply(act(a, p, g), b)
        case Product(a, b):
            return multiply(a, act(b, p - _arity(a), g))
        case _ if _retained(f):
            return Action(f, p, g)
        case _ if f.commuting:
            return _commute(f, lambda b: act(b, p, g), _members(g))
        case Zero(s, _, args):
            return Zero(
                s,
                _free_indices(f, *_members(g)),
                _without(args, p),
                f.scalar_type,
            )
        case _:
            raise TypeError(f"cannot occur: {f}")


def _bind(
    form: Form, q: int, g: AbstractExpression | tuple[AbstractExpression, ...]
) -> AbstractExpression:
    rest = _freshen(Form(form.expression, _without(form.arguments, q)), _members(g))
    pairs = zip(_members(form.arguments[q]), _members(g), strict=True)
    body = cast(
        "AbstractExpression",
        _substitute(rest.expression, dict(_binding(entry, value) for entry, value in pairs)),
    )
    return Form(body, rest.arguments) if rest.arguments else body


def _binding(
    entry: Function | Conjugate, value: AbstractExpression
) -> tuple[Function, AbstractExpression]:
    # The value is that of the entry, so v in an entry conjugate(v) takes the conjugate.
    match entry:
        case Conjugate(v):
            return _entry_function(entry), Conjugate(_promote(v, value))
        case _:
            return entry, _promote(entry, value)


def _without(xs: tuple[_T, ...], i: int) -> tuple[_T, ...]:
    return xs[:i] + xs[i + 1 :]


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


def _retained(form: AbstractExpression) -> bool:
    # Spatial operations transform the output; applying them must not move the
    # argument inside again. Further partial applications remain nested.
    return isinstance(form, Function | Gradient | PullBack | Action) or _field_terminal(form)


def _actions_complete(e: AbstractExpression) -> bool:
    return _everywhere(e, lambda n: not isinstance(n, Action) or _retained(n.form))
