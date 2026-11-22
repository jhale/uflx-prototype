import basix

from uflx.arithmetic import num
from uflx.calculus import coordinate, gradient, integral
from uflx.forms import function
from uflx.geometry import mesh
from uflx.nodes import EuclideanSpace
from uflx.tensors import tensor
from uflx_basix.basix_cells import basix_cell
from uflx_basix.basix_spaces import basix_space

from .meshes import grouped_topology, push_forward, side_domain

P, RT = basix.ElementFamily.P, basix.ElementFamily.RT
K, I = basix_cell(basix.CellType.triangle), basix_cell(basix.CellType.interval)
T = grouped_topology((K,), (((0, 1, 2), (0, 2, 3)),), ())
BOUNDARY = grouped_topology((I,), (((0, 1), (1, 2), (2, 3), (0, 3)),), ((("inside", T, 0),),))
X = basix_space("X", T, P, 1)
x0, x1 = function(X, "x0"), function(X, "x1")
v0, v1 = function(X, "v0"), function(X, "v1")
D = mesh(tensor((x0, x1)), EuclideanSpace(2))


def functionals():
    u = push_forward(function(basix_space("V", T, P, 2), "u"), D)
    w = push_forward(function(basix_space("R", T, RT, 1), "w"), D)
    g, y = gradient(u, D), coordinate(D)
    return {
        "area": integral(num(1), D),
        "energy": integral(g[0] * g[0] + g[1] * g[1], D),
        "field in the coordinates": integral(y[0] * y[0] * y[1], D),
        "perimeter": integral(num(1), side_domain(D, BOUNDARY)),
        "mass of a Raviart-Thomas function": integral(w[0] * w[0] + w[1] * w[1], D),
    }
