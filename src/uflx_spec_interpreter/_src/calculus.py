from .nodes import Coordinate, DirectionalDerivative, Gradient, Integral


def coordinate(domain):
    return Coordinate(domain)


def gradient(expression, domain):
    return Gradient(expression, domain)


def directional_derivative(expression, variable, direction, point=None):
    # The point is the variable unless given, as for the free function of the specification.
    return DirectionalDerivative(
        expression, variable, variable if point is None else point, direction
    )


def integral(expression, domain):
    return Integral(expression, domain)
