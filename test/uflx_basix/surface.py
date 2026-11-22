from uflx.geometry import mesh
from uflx.nodes import EuclideanSpace
from uflx_basix.nodes import VertexCoordinates

from .integrals import SQUARE

R3 = EuclideanSpace(3)

# A surface folded along its diagonal: two triangles that are not coplanar.
FOLD = VertexCoordinates(
    SQUARE, ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.5), (0.0, 1.0, 1.0))
)
M = mesh(FOLD, R3)
