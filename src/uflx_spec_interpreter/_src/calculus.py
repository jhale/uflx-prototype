from .nodes import Coordinate, DirectionalDerivative, Gradient, Integral


def coordinate(P):
    return Coordinate(P)


def gradient(f, P):
    return Gradient(f, P)


def directional_derivative(f, x, a, h):
    return DirectionalDerivative(f, x, a, h)


def integral(f, P):
    return Integral(f, P)
