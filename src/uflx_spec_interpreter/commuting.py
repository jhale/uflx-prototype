from .nodes import (
    CT,
    Conj,
    D,
    Div,
    Form,
    Gradient,
    Im,
    Indexed,
    Integral,
    ISum,
    PullBack,
    Re,
    Sum,
    Tensor,
)
from .substitution import freshen, map_fields

# The commuting node classes and the operands into which an evaluated node moves.
COMMUTING = {
    Sum: ("a", "b"),
    Tensor: ("entries",),
    Indexed: ("e",),
    CT: ("e",),
    ISum: ("e",),
    Form: ("e",),
    D: ("f",),
    Conj: ("a",),
    Re: ("a",),
    Im: ("a",),
    Div: ("a",),
    Gradient: ("b",),
    PullBack: ("b",),
    Integral: ("b",),
}


def commute(f, evaluate, avoid):
    # Move an evaluated node into the listed operands, after making a binding node fresh.
    fresh = freshen(f, avoid) if f.scope else f
    return map_fields(fresh, evaluate, COMMUTING[type(fresh)])
