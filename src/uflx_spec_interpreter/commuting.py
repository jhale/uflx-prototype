from .nodes import (
    ComponentTensor,
    Conjugate,
    DirectionalDerivative,
    Division,
    Form,
    Gradient,
    ImaginaryPart,
    Indexed,
    IndexSum,
    Integral,
    PullBack,
    RealPart,
    Sum,
    Tensor,
)
from .substitution import freshen, map_fields

# The commuting node classes and the operands into which an evaluated node moves.
COMMUTING = {
    Sum: ("a", "b"),
    Tensor: ("entries",),
    Indexed: ("e",),
    ComponentTensor: ("e",),
    IndexSum: ("e",),
    Form: ("e",),
    DirectionalDerivative: ("f",),
    Conjugate: ("a",),
    RealPart: ("a",),
    ImaginaryPart: ("a",),
    Division: ("a",),
    Gradient: ("b",),
    PullBack: ("b",),
    Integral: ("b",),
}


def commute(f, evaluate, avoid):
    # Move an evaluated node into the listed operands, after making a binding node fresh.
    fresh = freshen(f, avoid) if f.scope else f
    return map_fields(fresh, evaluate, COMMUTING[type(fresh)])
