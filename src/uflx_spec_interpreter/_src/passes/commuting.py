from ..nodes import (
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
    PushForward,
    RealPart,
    Sum,
    Tensor,
)
from ..substitution import freshen, map_fields

# The commuting node classes and the operands into which an evaluated node moves.
COMMUTING = {
    Sum: ("first", "second"),
    Tensor: ("entries",),
    Indexed: ("expression",),
    ComponentTensor: ("expression",),
    IndexSum: ("expression",),
    Form: ("expression",),
    DirectionalDerivative: ("expression",),
    Conjugate: ("operand",),
    RealPart: ("operand",),
    ImaginaryPart: ("operand",),
    Division: ("first",),
    Gradient: ("expression",),
    PullBack: ("expression",),
    PushForward: ("expression",),
    Integral: ("expression",),
}


def commute(f, evaluate, avoid):
    # Move an evaluated node into the listed operands, after making a binding node fresh.
    fresh = freshen(f, avoid)
    return map_fields(fresh, evaluate, COMMUTING[type(fresh)])
