from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar, cast

from ..nodes import Action, Conjugate, Form, Function, Gradient, Product, PullBack, Zero
from ..nodes.forms import entry_function
from ..static import arity, complete_indices, field_terminal, free_indices, promote
from ..substitution import everywhere, freshen, lowered, map_values, memoized, substitute
from ..tuples import members
from .commuting import commute

if TYPE_CHECKING:
    from ..nodes.base import AbstractExpression


T = TypeVar("T")


def rule(
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
            from ..tuples import each

            return Conjugate(act(e, p, each(Conjugate, g)))
        case Form(e, _) if p >= arity(e):
            return bind(f, p - arity(e), g)
        case Product(a, b) if p < arity(a):
            return Product(act(a, p, g), b)
        case Product(a, b):
            return Product(a, act(b, p - arity(a), g))
        case _ if retained(f):
            return Action(f, p, g)
        case _ if f.commuting:
            return commute(f, lambda b: act(b, p, g), members(g))
        case Zero(s, _, args):
            return Zero(
                s,
                free_indices(f, *members(g)),
                without(args, p),
                f.scalar_type,
            )
        case _:
            raise TypeError(f"cannot occur: {f}")


def bind(
    form: Form, q: int, g: AbstractExpression | tuple[AbstractExpression, ...]
) -> AbstractExpression:
    rest = freshen(Form(form.expression, without(form.arguments, q)), members(g))
    pairs = zip(members(form.arguments[q]), members(g), strict=True)
    body = cast(
        "AbstractExpression",
        substitute(rest.expression, dict(binding(entry, value) for entry, value in pairs)),
    )
    return Form(body, rest.arguments) if rest.arguments else body


def binding(
    entry: Function | Conjugate, value: AbstractExpression
) -> tuple[Function, AbstractExpression]:
    # The value is that of the entry, so v in an entry conjugate(v) takes the conjugate.
    match entry:
        case Conjugate(v):
            return entry_function(entry), Conjugate(promote(v, value))
        case _:
            return entry, promote(entry, value)


def without(xs: tuple[T, ...], i: int) -> tuple[T, ...]:
    return xs[:i] + xs[i + 1 :]


def apply_actions(f: AbstractExpression) -> AbstractExpression:
    # Inner actions first, then the rule for the form of each action.
    act = memoized(rule)

    def apply(
        walk: Callable[[AbstractExpression], AbstractExpression], e: AbstractExpression
    ) -> AbstractExpression:
        e = map_values(e, walk, lowered(e))
        match e:
            case Action(form, p, g):
                out = complete_indices(e, act(form, p, g))
                return out if out is e else walk(out)
            case _:
                return e

    return memoized(apply)(f)


def retained(form: AbstractExpression) -> bool:
    # Spatial operations transform the output; applying them must not move the
    # argument inside again. Further partial applications remain nested.
    return isinstance(form, Function | Gradient | PullBack | Action) or field_terminal(form)


def in_n2(e: AbstractExpression) -> bool:
    return everywhere(e, lambda n: not isinstance(n, Action) or retained(n.form))
