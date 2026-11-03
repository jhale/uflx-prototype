from .names import derived_indices
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
    *lead, k = derived_indices("uflx.divergence", (f, P), len(f.value_shape))
    entry = IndexSum(Indexed(Gradient(f, P), (*lead, k, k)), (k,))
    return ComponentTensor(entry, tuple(lead)) if lead else entry


def rotation_of_gradient(f, P):
    # The value of Curl(f, P), the sum of epsilon[..., j, k] * grad(f)[k, j] over j and k, with
    # the permutation tensor of rank n, so a vector for n = 3 and a scalar for n = 2.
    n = P.ambient_coordinate_domain.dimension
    *out, j, k = derived_indices("uflx.curl", (f, P), n)
    entry = IndexSum(
        Product(Indexed(permutation_tensor(n), (*out, j, k)), Indexed(Gradient(f, P), (k, j))),
        (j, k),
    )
    return ComponentTensor(entry, tuple(out)) if out else entry
