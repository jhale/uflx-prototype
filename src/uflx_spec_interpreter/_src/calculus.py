from .arithmetic import derived
from .nodes import (
    ComponentTensor,
    Coordinate,
    Curl,
    DirectionalDerivative,
    Divergence,
    Gradient,
    Indexed,
    IndexSum,
    Integral,
    Product,
    PushForward,
    TransformationLaw,
)
from .tensors import permutation_tensor


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
    *lead, k = derived("uflx.divergence", (f, P), len(f.value_shape))
    entry = IndexSum(Indexed(Gradient(f, P), (*lead, k, k)), (k,))
    return ComponentTensor(entry, tuple(lead)) if lead else entry


def rotation_of_gradient(f, P):
    # The value of Curl(f, P), the sum of epsilon[..., j, k] * grad(f)[k, j] over j and k, with
    # the permutation tensor of rank n, so a vector for n = 3 and a scalar for n = 2.
    n = P.ambient_coordinate_domain.dimension
    *out, j, k = derived("uflx.curl", (f, P), n)
    entry = IndexSum(
        Product(Indexed(permutation_tensor(n), (*out, j, k)), Indexed(Gradient(f, P), (k, j))),
        (j, k),
    )
    return ComponentTensor(entry, tuple(out)) if out else entry


# The value of each operator as a composition of the gradient.
VALUE = {Divergence: trace_of_gradient, Curl: rotation_of_gradient}

# The law of a push-forward whose divergence or curl is the operator on the reference cell.
PIOLA = {
    Divergence: TransformationLaw(("contravariant",), 1),
    Curl: TransformationLaw(("covariant",), 0),
}


def piola(e):
    # The identity of the divergence holds on every mesh, and that of the curl for k = n only.
    f, P = e.expression, e.domain
    return (
        isinstance(f, PushForward)
        and f.domain == P
        and f.law == PIOLA[type(e)]
        and (isinstance(e, Divergence) or P.dimension == P.ambient_coordinate_domain.dimension)
    )


def result_law(e):
    # The divergence or the curl of a push-forward of Piola is a density of weight 1, with a
    # contravariant axis for each axis that the operator makes and the leading axes unchanged.
    f = e.expression
    lead = len(f.value_shape) - len(f.law.variances)
    return TransformationLaw(("contravariant",) * (len(e.value_shape) - lead), 1)
