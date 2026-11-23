import basix

from uflx.arithmetic import num
from uflx.calculus import coordinate, gradient, integral
from uflx.forms import function
from uflx.geometry import mesh
from uflx.nodes import EuclideanSpace
from uflx.tensors import tensor
from uflx_basix.cells import cell
from uflx_basix.spaces import space

from .meshes import grouped_topology, push_forward, side_domain

P, RT = basix.ElementFamily.P, basix.ElementFamily.RT
K, I = cell(basix.CellType.triangle), cell(basix.CellType.interval)
T = grouped_topology((K,), (((0, 1, 2), (0, 2, 3)),), ())
BOUNDARY = grouped_topology((I,), (((0, 1), (1, 2), (2, 3), (0, 3)),), ((("inside", T, 0),),))
X = space("X", basix.create_element(P, K.cell_type, 1), domain=T)
x0, x1 = function(X, "x0"), function(X, "x1")
v0, v1 = function(X, "v0"), function(X, "v1")
D = mesh(tensor((x0, x1)), EuclideanSpace(2))


def functionals():
    u = push_forward(
        function(space("V", basix.create_element(P, K.cell_type, 2), domain=T), "u"), D
    )
    w = push_forward(
        function(space("R", basix.create_element(RT, K.cell_type, 1), domain=T), "w"), D
    )
    g, y = gradient(u, D), coordinate(D)
    return {
        "area": integral(num(1), D),
        "energy": integral(g[0] * g[0] + g[1] * g[1], D),
        "field in the coordinates": integral(y[0] * y[0] * y[1], D),
        "perimeter": integral(num(1), side_domain(D, BOUNDARY)),
        "mass of a Raviart-Thomas function": integral(w[0] * w[0] + w[1] * w[1], D),
    }
