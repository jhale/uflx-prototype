from .arithmetic import add, subtract
from .nodes import Absolute, Coordinate, Gradient, Integer, Inverse, SquareRoot
from .tensors import determinant, identity, matrix_product, transpose


def gram(J):
    return matrix_product(transpose(J), J)


def tangent_gram(J, P):
    # JᵀJ + I - Π is invertible on a parameter domain of lower dimension, where JᵀJ is not.
    identity_matrix = identity(P.ambient_coordinate_domain.dimension)
    return subtract(add(gram(J), identity_matrix), Gradient(Coordinate(P), P))


def jacobian(m):
    return Gradient(m.parameterisation, m.parameter_domain)


def metric(m):
    # JᵀJ + I - Π, which is JᵀJ when the parameter domain has the dimension of its ambient.
    J = jacobian(m)
    return (
        gram(J)
        if m.parameter_domain.dimension == m.parameter_domain.ambient_coordinate_domain.dimension
        else tangent_gram(J, m.parameter_domain)
    )


def density(m):
    # sqrt(det G), with the closed form |det J| for a cell of the ambient dimension.
    k, n = m.parameter_domain.dimension, m.ambient_domain.ambient_coordinate_domain.dimension
    if k == 0:
        return Integer(1)
    if k == m.parameter_domain.ambient_coordinate_domain.dimension == n:
        return Absolute(determinant(jacobian(m), k))
    return SquareRoot(
        determinant(metric(m), m.parameter_domain.ambient_coordinate_domain.dimension)
    )


def inverse_jacobian(m):
    # G⁻¹Jᵀ, with the closed form inverse(J) when the parameter domain has the ambient dimension.
    if m.parameter_domain.dimension == m.parameter_domain.ambient_coordinate_domain.dimension:
        return Inverse(jacobian(m))
    return matrix_product(Inverse(metric(m)), transpose(jacobian(m)))
