from .arithmetic import add, subtract
from .nodes import Absolute, Coordinate, Gradient, Integer, Inverse, SquareRoot
from .tensors import determinant, identity, matrix_product, transpose


def gram(J):
    return matrix_product(transpose(J), J)


def tangent_gram(J, P):
    # JᵀJ + I - Π is invertible on a parameter domain of lower dimension, where JᵀJ is not.
    identity_matrix = identity(P.ambient_coordinate_domain.dimension)
    return subtract(add(gram(J), identity_matrix), Gradient(Coordinate(P), P))


def jacobian(D):
    return Gradient(D.parameterisation, D.parameter_domain)


def metric(D):
    # JᵀJ + I - Π, which is JᵀJ when the parameter domain has the dimension of its ambient.
    J = jacobian(D)
    return (
        gram(J)
        if D.parameter_domain.dimension == D.parameter_domain.ambient_coordinate_domain.dimension
        else tangent_gram(J, D.parameter_domain)
    )


def density(D):
    # sqrt(det G), with the closed form |det J| for a cell of the ambient dimension.
    k, n = D.parameter_domain.dimension, D.ambient_domain.ambient_coordinate_domain.dimension
    if k == 0:
        return Integer(1)
    if k == D.parameter_domain.ambient_coordinate_domain.dimension == n:
        return Absolute(determinant(jacobian(D)))
    return SquareRoot(determinant(metric(D)))


def inverse_jacobian(D):
    # G⁻¹Jᵀ, with the closed form inverse(J) when the parameter domain has the ambient dimension.
    if D.parameter_domain.dimension == D.parameter_domain.ambient_coordinate_domain.dimension:
        return Inverse(jacobian(D))
    return matrix_product(Inverse(metric(D)), transpose(jacobian(D)))
