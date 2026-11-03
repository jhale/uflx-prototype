from ..nodes import DirectionalDerivative
from ..static import complete_indices, promote, zero_like
from ..substitution import bottom_up, everywhere, memoized, occurs, substitute_values
from ..tuples import members
from .actions import apply_actions
from .derivative_table import derivative_of_node


def deriv(f, x, a, h):
    # Evaluate undifferentiated operands at the point; inserted directions stay unchanged.
    xs, points, directions = members(x), members(a), members(h)
    points = tuple(promote(v, p) for v, p in zip(xs, points, strict=True))
    directions = tuple(promote(v, w) for v, w in zip(xs, directions, strict=True))
    mapping = dict(zip(xs, points, strict=True))
    at = memoized(lambda _, e: substitute_values(e, mapping))
    constant = lambda e: not any(occurs(v, e) for v in xs)

    def differentiate(d, e):
        if e in xs:
            return directions[xs.index(e)]
        if constant(e):
            return zero_like(e, e.argument_spaces, *points, *directions)
        return derivative_of_node(e, d, constant, [*xs, *points, *directions], at)

    return complete_indices(DirectionalDerivative(f, x, a, h), memoized(differentiate)(f))


def apply_derivatives(f):
    # Derivative evaluation does no optional algebraic collection.
    differential = memoized(lambda _, g, x, a, h: deriv(g, x, a, h))

    def evaluate(e):
        match e:
            case DirectionalDerivative(g, x, a, h):
                return differential(g, x, a, h)
        return e

    # Applying a newly exposed action can reveal another derivative.
    out = apply_actions(bottom_up(f, evaluate))
    return out if in_n3(out) else apply_derivatives(out)


def in_n3(e):
    return everywhere(e, lambda n: not isinstance(n, DirectionalDerivative))
