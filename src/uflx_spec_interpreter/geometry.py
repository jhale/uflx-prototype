from .algebra import determinant, gram, matmul, tangent_gram, transpose
from .domains import ambient_dim, topdim
from .nodes import Abs, Gradient, Inv, Num, Sqrt


def jacobian(m):
    return Gradient(m.psi, m.P)


def metric(m):
    # JᵀJ + I - Π, which is JᵀJ when the parameter domain has the dimension of its ambient.
    J = jacobian(m)
    return gram(J) if topdim(m.P) == ambient_dim(m.P) else tangent_gram(J, m.P)


def density(m):
    # sqrt(det G), with the closed form |det J| for a cell of the ambient dimension.
    k, n = topdim(m.P), ambient_dim(m.Q)
    if k == 0:
        return Num(1)
    if k == ambient_dim(m.P) == n:
        return Abs(determinant(jacobian(m), k))
    return Sqrt(determinant(metric(m), ambient_dim(m.P)))


def inverse_jacobian(m):
    # G⁻¹Jᵀ, with the closed form inverse(J) when the parameter domain has the ambient dimension.
    if topdim(m.P) == ambient_dim(m.P):
        return Inv(jacobian(m))
    return matmul(Inv(metric(m)), transpose(jacobian(m)))
