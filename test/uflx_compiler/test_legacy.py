import basix
import numpy as np

from uflx._src.nodes import Integer, Product, Sum
from uflx.arithmetic import conjugate
from uflx.calculus import gradient, integral
from uflx.forms import form, function
from uflx.geometry import pull_back
from uflx.tensors import index, index_sum
from uflx_basix.assembly import conforming_dofmap
from uflx_basix.basix_spaces import basix_space
from uflx_basix.sides import outward_conormal
from uflx_compiler import assemble_bound

from ..uflx_basix.legacy import Constant
from ..uflx_basix.meshes import push_forward, side_domain, side_map
from ..uflx_basix.test_legacy import GEOMETRY, LEFT, LOWER, D, T, demo

RNG = np.random.default_rng(5)
DATA = GEOMETRY | {
    "coefficients:kappa": np.array(1.7),
    "coefficients:f": RNG.normal(size=4),
    "coefficients:('q', 0)": RNG.normal(size=4),
    "coefficients:('q', 1)": RNG.normal(size=4),
}


def core():
    # The same forms written with the core alone, each push-forward explicit.
    V = basix_space("V", T, basix.ElementFamily.P, 1)
    u, v = function(V, ("argument", 1)), function(V, ("argument", 0))
    f, q0, q1 = function(V, "f"), function(V, ("q", 0)), function(V, ("q", 1))
    kappa = Constant("kappa")
    uD, vD, fD = (push_forward(w, D) for w in (u, v, f))
    qD = (push_forward(q0, D), push_forward(q1, D))
    i = index("i")
    gu, gv = gradient(uD, D), gradient(vD, D)
    a = form(integral(kappa * index_sum(gu[i] * conjugate(gv[i]), (i,)), D), (conjugate(v), u))
    div_q = gradient(qD[0], D)[0] + gradient(qD[1], D)[1]
    lower, left = (
        side_domain(D, LOWER),
        side_domain(D, LEFT),
    )
    on_lower, on_left = side_map(D, lower, "inside"), side_map(D, left, "inside")
    n = outward_conormal(on_left)
    flux = pull_back(qD[0], on_left) * n[0] + pull_back(qD[1], on_left) * n[1]
    L = (
        form(integral(pull_back(fD * conjugate(vD), on_lower), lower), (conjugate(v),))
        - form(integral(div_q * conjugate(vD), D), (conjugate(v),))
        + form(integral(flux * conjugate(pull_back(vD, on_left)), left), (conjugate(v),))
    )
    return a, L


def assembled(f):
    # The toy assembles one form, so a sum of forms is the sum of the assembled terms.
    if isinstance(f, Sum):
        return sum(assembled(g) for g in f.construction_arguments)
    if isinstance(f, Product) and isinstance(f.construction_arguments[0], Integer):
        c, g = f.construction_arguments
        return c.value * assembled(g)
    from uflx._src.algorithms.traversal import everywhere
    from uflx._src.nodes import Function
    from uflx_basix.basix_nodes import BasixSpace

    bindings = {}

    def bind(node):
        if (
            isinstance(node, Function)
            and isinstance(node.space, BasixSpace)
            and node is not D.coordinate_function
        ):
            bindings[node] = conforming_dofmap(node.space)
        return True

    everywhere(f, bind)
    return assemble_bound(f, 4, bindings)(DATA)


def test_the_classic_poisson_script_gives_the_forms_of_the_core():
    for layer, by_hand in zip(demo(), core(), strict=True):
        assert np.allclose(assembled(layer), assembled(by_hand))
    # The rows of a stiffness matrix sum to zero, because the constants are in its kernel.
    assert np.allclose(assembled(demo()[0]).sum(axis=1), 0)
