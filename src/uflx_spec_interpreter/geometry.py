from .algebra import determinant, gram, matrix_product, tangent_gram, transpose
from .nodes import Absolute, Gradient, Inverse, Num, SquareRoot


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
        return Num(1)
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
