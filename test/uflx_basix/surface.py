from uflx.nodes import EuclideanSpace

from .integrals import SQUARE
from .meshes import vertex_mesh

R3 = EuclideanSpace(3)

# A surface folded along its diagonal: two triangles that are not coplanar.
M, GEOMETRY = vertex_mesh(
    SQUARE, ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.5), (0.0, 1.0, 1.0)), label="fold"
)
FOLD = M.coordinate_function
