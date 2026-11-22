from uflx.forms import function
from uflx.tensors import index
from uflx_basix.cells import reference_simplex
from uflx_basix.spaces import space

from .meshes import grouped_topology, vertex_mesh

K = reference_simplex(2)
# The cells of the square, which the folded surface of test_surface shares.
SQUARE = grouped_topology((K,), (((0, 1, 2), (0, 2, 3)),), ())
I = reference_simplex(1)
# The diagonal, seen from both triangles, and the boundary, seen from inside.
DIAGONAL = grouped_topology((I,), (((0, 2),),), ((("+", SQUARE, 0), ("-", SQUARE, 0)),))
c = index("load-entry")

# The square as a mesh of the coordinates of its vertices, with the cell index c of SQUARE.
D, GEOMETRY = vertex_mesh(SQUARE, [(0, 0), (1, 0), (1, 1), (0, 1)])

V = space("V", domain=D)
u = function(V, "u")
