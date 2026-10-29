from .nodes import (
    ComponentTensor,
    Coordinate,
    Curl,
    DirectionalDerivative,
    Divergence,
    Gradient,
    Index,
    Indexed,
    IndexSum,
    Integral,
    PushForward,
    TransformationLaw,
)
from .tensors import tensor


def coordinate(P):
    return Coordinate(P)


def gradient(f, P):
    return Gradient(f, P)


def divergence(f, P):
    return Divergence(f, P)


def curl(f, P):
    return Curl(f, P)


def directional_derivative(f, x, a, h):
    return DirectionalDerivative(f, x, a, h)


def integral(f, P):
    return Integral(f, P)


def trace_of_gradient(f, P):
    # The value of Divergence(f, P), the last axis of f contracted with the axis of the gradient.
    lead = tuple(Index(("uflx.divergence", f, P, n)) for n in range(len(f.value_shape) - 1))
    k = Index(("uflx.divergence", f, P, "k"))
    entry = IndexSum(Indexed(Gradient(f, P), (*lead, k, k)), (k,))
    return ComponentTensor(entry, lead) if lead else entry


def rotation_of_gradient(f, P):
    # The value of Curl(f, P), from the entries of the gradient.
    G = Gradient(f, P)
    if f.value_shape == (3,):
        return tensor((G[2, 1] - G[1, 2], G[0, 2] - G[2, 0], G[1, 0] - G[0, 1]))
    return G[1, 0] - G[0, 1]


CONTRAVARIANT = TransformationLaw(("contravariant",), 1)
COVARIANT = TransformationLaw(("covariant",), 0)


def piola(e):
    # Whether the divergence or the curl of a push-forward to P is the operator on the reference
    # cell carried by a law, with no derivative of the Jacobian.
    f, P = e.expression, e.domain
    if not isinstance(f, PushForward) or f.domain != P:
        return False
    if isinstance(e, Divergence):
        return f.law == CONTRAVARIANT
    n = P.ambient_coordinate_domain.dimension
    return (
        f.law == COVARIANT and f.value_shape == (n,) and P.pieces[0].parameter_domain.dimension == n
    )
