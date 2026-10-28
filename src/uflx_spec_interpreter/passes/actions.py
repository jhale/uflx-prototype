from ..forms import members
from ..nodes import Action, Conjugate, Form, Function, Product, Zero
from ..static import arity, free_indices
from ..substitution import expressions, freshen, map_fields, substitute
from .commuting import COMMUTING, commute


def act(f, p, g):
    # The rule table of apply_actions, keyed on the form f.
    match f:
        case Form(e, _) if p >= arity(e):
            return bind(f, p - arity(e), g)
        case Product(a, b) if p < arity(a):
            return Product(act(a, p, g), b)
        case Product(a, b):
            return Product(a, act(b, p - arity(a), g))
        case _ if type(f) in COMMUTING:
            return commute(f, lambda b: act(b, p, g), members(g))
        case Zero(s, _, args):
            return Zero(s, free_indices(f, *members(g)), without(args, p))
        case Function(space, _) if space.dual:
            return Action(f, p, g)
    raise TypeError(f"cannot occur: {f}")


def bind(form, q, g):
    rest = freshen(form.rebuild(arguments=without(form.arguments, q)), members(g))
    pairs = zip(members(form.arguments[q]), members(g), strict=True)
    body = substitute(rest.expression, dict(binding(entry, value) for entry, value in pairs))
    return rest.rebuild(expression=body) if rest.arguments else body


def binding(entry, value):
    # The value is that of the entry, so v in an entry conjugate(v) takes the conjugate.
    match entry:
        case Conjugate(v):
            return v, Conjugate(value)
    return entry, value


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
        case Action(Function(space, _), _, _) if space.dual:
            pass
        case Action():
            return False
    return all(in_n2(c) for c in expressions(e))
