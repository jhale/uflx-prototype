from .nodes import Absolute, Gradient, Integer, Inverse, SquareRoot
from .tensors import determinant, matrix_product, transpose


def jacobian(D):
    return Gradient(D.parameterisation, D.parameter_domain)


def metric(D):
    # JᵀJ, the metric that the parameterisation induces on its parameter domain.
    J = jacobian(D)
    return matrix_product(transpose(J), J)


def density(D):
    # sqrt(det G), with the closed form |det J| for a cell of the ambient dimension.
    k, n = D.parameter_domain.dimension, D.ambient_domain.ambient_coordinate_domain.dimension
    if k == 0:
        return Integer(1)
    if k == n:
        return Absolute(determinant(jacobian(D)))
    return SquareRoot(determinant(metric(D)))


def inverse_jacobian(D):
    # The pseudo-inverse of J, which is the inverse for a cell of the ambient dimension.
    return Inverse(jacobian(D))
