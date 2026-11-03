from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar, cast

from ..arithmetic import multiply
from ..nodes import Action, Conjugate, Form, Function, Gradient, Product, PullBack, Zero
from ..nodes.forms import _entry_function
from ..static import _arity, _complete_indices, _field_terminal, _free_indices, _promote
from ..substitution import _everywhere, _freshen, _lowered, _map_values, _memoized, _substitute
from ..tuples import _members
from .commuting import _commute

if TYPE_CHECKING:
    from ..nodes.base import AbstractExpression


_T = TypeVar("_T")


# Shared implementation helpers; the package public API is declared in __init__.py.
__all__ = ["_in_n2", "_retained", "apply_actions"]


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

    def apply(
        walk: Callable[[AbstractExpression], AbstractExpression], e: AbstractExpression
    ) -> AbstractExpression:
        e = _map_values(e, walk, _lowered(e))
        match e:
            case Action(form, p, g):
                out = _complete_indices(e, act(form, p, g))
                return out if out is e else walk(out)
            case _:
                return e

    return _memoized(apply)(f)


def _retained(form: AbstractExpression) -> bool:
    # Spatial operations transform the output; applying them must not move the
    # argument inside again. Further partial applications remain nested.
    return isinstance(form, Function | Gradient | PullBack | Action) or _field_terminal(form)


def _in_n2(e: AbstractExpression) -> bool:
    return _everywhere(e, lambda n: not isinstance(n, Action) or _retained(n.form))
