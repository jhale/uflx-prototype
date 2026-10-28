from dataclasses import replace

from core import COMMUTING, Action, Fn, Form, Prod, Zero, commute, entry_fns, freshen, map_fields, members, operands, substitute
from static import arity, free_indices


def act(f, p, g):
    # The rule table of apply_actions, keyed on the form f.
    match f:
        case Form(e, _) if p >= arity(e):
            return bind(f, p - arity(e), g)
        case Prod(a, b) if p < arity(a):
            return Prod(act(a, p, g), b)
        case Prod(a, b):
            return Prod(a, act(b, p - arity(a), g))
        case _ if type(f) in COMMUTING:
            return commute(f, lambda b: act(b, p, g), members(g))
        case Zero(s, _, args):
            return Zero(s, free_indices(f, *members(g)), without(args, p))
        case Fn(space, _) if space.dual:
            return Action(f, p, g)
    raise TypeError(f"cannot occur: {f}")


def bind(form, q, g):
    rest = freshen(replace(form, args=without(form.args, q)), members(g))
    body = substitute(rest.e, dict(zip(entry_fns(form.args[q]), members(g))))
    return replace(rest, e=body) if rest.args else body


def without(xs, i):
    return xs[:i] + xs[i + 1 :]


def apply_actions(e):
    # Inner actions first, then the rule for the form of each action.
    match map_fields(e, apply_actions):
        case Action(f, p, g):
            return act(f, p, g)
        case e:
            return e


def in_n2(e):
    match e:
        case Action(Fn(space, _), _, _) if space.dual:
            pass
        case Action():
            return False
    return all(in_n2(c) for c in operands(e))
