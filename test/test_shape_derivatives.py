import basix
import numpy as np

from uflx_compiler import assemble, compile_form, evaluate
from uflx_spec_interpreter import (
    RD,
    coordinate,
    form,
    function,
    gradient,
    index,
    index_sum,
    integral,
    num,
    side_domain,
    tensor,
)
from uflx_spec_interpreter._draft import image, push_forward, shape_derivative
from uflx_toy.basix_cells import basix_cell
from uflx_toy.basix_spaces import basix_space
from uflx_toy.topology import static_topology

P, RT = basix.ElementFamily.P, basix.ElementFamily.RT
K, I = basix_cell(basix.CellType.triangle), basix_cell(basix.CellType.interval)
T = static_topology((K,), (((0, 1, 2), (0, 2, 3)),), ())
BOUNDARY = static_topology((I,), (((0, 1), (1, 2), (2, 3), (0, 3)),), ((("inside", T, 0),),))
X = basix_space("X", T, P, 1)
x0, x1 = function(X, "x0"), function(X, "x1")
v0, v1 = function(X, "v0"), function(X, "v1")
D = image(tensor((x0, x1)), RD(2))
# The dofs of P1 on T are the vertices, so the coordinates are those of a skewed square.
COORDINATES = np.array([(0.0, 0.0), (1.0, 0.1), (1.2, 1.1), (-0.1, 0.9)])
RNG = np.random.default_rng(3)
DIRECTION = RNG.normal(size=(4, 2))
U = RNG.normal(size=basix_space("V", T, P, 2).num_dofs)
W = RNG.normal(size=basix_space("R", T, RT, 1).num_dofs)


def env(coordinates):
    return {
        "coefficients:x0": coordinates[:, 0],
        "coefficients:x1": coordinates[:, 1],
        "coefficients:v0": DIRECTION[:, 0],
        "coefficients:v1": DIRECTION[:, 1],
        "coefficients:u": U,
        "coefficients:w": W,
    }


def value(f, coordinates):
    return evaluate(compile_form(f, 8), env(coordinates))


def difference(f, step=1e-6):
    # The central difference of f with the coordinates moved along the direction.
    plus, minus = COORDINATES + step * DIRECTION, COORDINATES - step * DIRECTION
    return (value(f, plus) - value(f, minus)) / (2 * step)


def shape(f):
    return value(shape_derivative(f, (x0, x1), (v0, v1)), COORDINATES)


def functionals():
    u = push_forward(function(basix_space("V", T, P, 2), "u"), D)
    w = push_forward(function(basix_space("R", T, RT, 1), "w"), D)
    g, y = gradient(u, D), coordinate(D)
    return {
        "area": integral(num(1), D),
        "energy": integral(g[0] * g[0] + g[1] * g[1], D),
        "field in the coordinates": integral(y[0] * y[0] * y[1], D),
        "perimeter": integral(num(1), side_domain(D, BOUNDARY, BOUNDARY.sides[0][0][0])),
        "mass of a Raviart-Thomas function": integral(w[0] * w[0] + w[1] * w[1], D),
    }


def test_shape_derivatives_agree_with_differences_of_the_coordinates():
    for name, f in functionals().items():
        assert np.isclose(shape(f), difference(f), rtol=1e-6, atol=1e-8), name


def test_the_shape_derivative_of_the_area_is_the_integral_of_the_divergence():
    i = index("i")
    V = tensor((push_forward(v0, D), push_forward(v1, D)))
    divergence = integral(index_sum(gradient(V, D)[i, i], (i,)), D)
    assert np.isclose(shape(integral(num(1), D)), value(divergence, COORDINATES))


def test_the_shape_gradient_assembles_to_the_derivative_in_each_dof():
    f = functionals()["energy"]
    z = function(X, "z")
    gradient_x0 = assemble(form(shape_derivative(f, (x0, x1), (z, num(0))), (z,)), 8)(
        env(COORDINATES)
    )
    step = 1e-6
    for dof in range(4):
        e = np.zeros((4, 2))
        e[dof, 0] = step
        central = (value(f, COORDINATES + e) - value(f, COORDINATES - e)) / (2 * step)
        assert np.isclose(gradient_x0[dof], central, rtol=1e-6, atol=1e-8), dof


def test_a_mesh_moved_by_a_displacement_is_the_mesh_of_the_moved_coordinates():
    # The coordinate function of an arbitrary Lagrangian-Eulerian mesh is x + d on the topology.
    d0, d1 = function(X, "d0"), function(X, "d1")
    moved = image(tensor((x0 + d0, x1 + d1)), RD(2))
    displacement = RNG.normal(size=(4, 2)) * 0.05
    data = env(COORDINATES) | {
        "coefficients:d0": displacement[:, 0],
        "coefficients:d1": displacement[:, 1],
    }
    for f, g in zip(area_and_energy(moved), area_and_energy(D), strict=True):
        assert np.isclose(evaluate(compile_form(f, 8), data), value(g, COORDINATES + displacement))


def area_and_energy(M):
    u = push_forward(function(basix_space("V", T, P, 2), "u"), M)
    g = gradient(u, M)
    return integral(num(1), M), integral(g[0] * g[0] + g[1] * g[1], M)
