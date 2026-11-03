from __future__ import annotations

from typing import TYPE_CHECKING

from ._src.calculus import curl, divergence
from ._src.domains import _cell_domain, _coordinate_function, cell_map, image, transformation_law
from ._src.domains import push_forward as _push_forward
from ._src.domains import side_domain as _side_domain
from ._src.domains import side_map as _side_map
from ._src.extensions.shape import _apply_shape_derivatives, _shape_derivative

if TYPE_CHECKING:
    from ._src.nodes.base import AbstractExpression
    from ._src.nodes.calculus import PushForward, TransformationLaw
    from ._src.nodes.domains import Mesh, SideMap
    from ._src.nodes.topology import AbstractTopology


# Private compatibility helpers for older interpreter examples and the shape extension.

__all__ = [
    "_apply_shape_derivatives",
    "_cell_domain",
    "_coordinate_function",
    "_shape_derivative",
    "cell_map",
    "curl",
    "divergence",
    "image",
    "push_forward",
    "side_map",
    "transformation_law",
]


def push_forward(f: AbstractExpression, D: Mesh, m: TransformationLaw | None = None) -> PushForward:
    from ._src.nodes.calculus import TransformationLaw
    from ._src.nodes.forms import Function

    if m is None:
        if not isinstance(f, Function):
            raise TypeError("an explicit law is required for this expression")
        law: object = getattr(f.space, "law", None)
        if not isinstance(law, TransformationLaw):
            raise TypeError("the function space supplies no transformation law")
        m = law
    return _push_forward(f, D, m)


def side_map(
    X: Mesh | AbstractTopology, P: Mesh | AbstractTopology, s: str | None = None
) -> SideMap:
    from ._src.nodes.domains import Mesh

    S = P.topology if isinstance(P, Mesh) else P
    return _side_map(X, P, S.sides[0][0][0] if s is None else s)


def side_domain(D: Mesh, S: AbstractTopology, s: str | None = None) -> Mesh:
    return _side_domain(D, S, S.sides[0][0][0] if s is None else s)
