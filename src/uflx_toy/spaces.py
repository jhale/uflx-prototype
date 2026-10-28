from .nodes import Space


def space(name, value_shape=(), domain=None, scalar_type="complex"):
    return Space(name, tuple(value_shape), domain, scalar_type)
