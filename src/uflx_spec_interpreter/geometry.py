from .algebra import determinant, gram, matrix_product, tangent_gram, transpose
from .domains import ambient_dim
from .nodes import Absolute, Gradient, Inverse, Num, SquareRoot


def jacobian(m):
    return Gradient(m.parameterisation, m.parameter_domain)


def metric(m):
    # JᵀJ + I - Π, which is JᵀJ when the parameter domain has the dimension of its ambient.
    J = jacobian(m)
    return (
        gram(J)
        if m.parameter_domain.dimension == ambient_dim(m.parameter_domain)
        else tangent_gram(J, m.parameter_domain)
    )


def density(m):
    # sqrt(det G), with the closed form |det J| for a cell of the ambient dimension.
    k, n = m.parameter_domain.dimension, ambient_dim(m.ambient_domain)
    if k == 0:
        return Num(1)
    if k == ambient_dim(m.parameter_domain) == n:
        return Absolute(determinant(jacobian(m), k))
    return SquareRoot(determinant(metric(m), ambient_dim(m.parameter_domain)))


def inverse_jacobian(m):
    # G⁻¹Jᵀ, with the closed form inverse(J) when the parameter domain has the ambient dimension.
    if m.parameter_domain.dimension == ambient_dim(m.parameter_domain):
        return Inverse(jacobian(m))
    return matrix_product(Inverse(metric(m)), transpose(jacobian(m)))
