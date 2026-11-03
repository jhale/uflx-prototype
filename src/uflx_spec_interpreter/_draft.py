# Private compatibility helpers for older interpreter examples and the shape extension.
from ._src.calculus import curl, divergence
from ._src.domains import (
    cell_domain,
    cell_map,
    coordinate_function,
    image,
    transformation_law,
)
from ._src.extensions.shape import apply_shape_derivatives, shape_derivative

__all__ = [
    "apply_shape_derivatives",
    "cell_domain",
    "cell_map",
    "coordinate_function",
    "curl",
    "divergence",
    "image",
    "push_forward",
    "shape_derivative",
    "side_map",
    "transformation_law",
]


from ._src.domains import push_forward as _push_forward
from ._src.domains import side_domain as _side_domain
from ._src.domains import side_map as _side_map


def push_forward(f, D, m=None):
    return _push_forward(f, D, f.space.law if m is None else m)


def side_map(X, P, s=None):
    S = P.topology if hasattr(P, "topology") else P
    return _side_map(X, P, S.sides[0][0][0] if s is None else s)


def side_domain(D, S, s=None):
    return _side_domain(D, S, S.sides[0][0][0] if s is None else s)
