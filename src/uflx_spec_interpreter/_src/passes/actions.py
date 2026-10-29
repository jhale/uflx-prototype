from ..nodes import Action, Conjugate, DualSpace, Form, Function, Product, Zero
from ..static import arity, free_indices
from ..substitution import bottom_up, everywhere, freshen, memoized, rebuild, substitute
from ..tuples import members
from .commuting import commute


def rule(act, f, p, g):
    # The rule table of apply_actions, keyed on the form f.
    match f:
        case Form(e, _) if p >= arity(e):
            return bind(f, p - arity(e), g)
        case Product(a, b) if p < arity(a):
            return Product(act(a, p, g), b)
        case Product(a, b):
            return Product(a, act(b, p - arity(a), g))
        case _ if f.commuting:
            return commute(f, lambda b: act(b, p, g), members(g))
        case Zero(s, _, args):
            return Zero(s, free_indices(f, *members(g)), without(args, p))
        case Function(space, _) if isinstance(space, DualSpace):
            return Action(f, p, g)
    raise TypeError(f"cannot occur: {f}")


def bind(form, q, g):
    rest = freshen(rebuild(form, arguments=without(form.arguments, q)), members(g))
    pairs = zip(members(form.arguments[q]), members(g), strict=True)
    body = substitute(rest.expression, dict(binding(entry, value) for entry, value in pairs))
    return rebuild(rest, expression=body) if rest.arguments else body


def binding(entry, value):
    # The value is that of the entry, so v in an entry conjugate(v) takes the conjugate.
    match entry:
        case Conjugate(v):
            return v, Conjugate(value)
    return entry, value


def without(xs, i):
    return xs[:i] + xs[i + 1 :]


def apply_actions(f):
    # Inner actions first, then the rule for the form of each action.
    act = memoized(rule)

    def apply(e):
        match e:
            case Action(form, p, g):
                return act(form, p, g)
        return e

    return bottom_up(f, apply)


def in_n2(e):
    def holds(n):
        match n:
            case Action(Function(space, _), _, _):
                return isinstance(space, DualSpace)
        return not isinstance(n, Action)

    return everywhere(e, holds)
