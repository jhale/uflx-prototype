from test_integrals import D, K, u

from uflx_spec_interpreter import (
    action,
    add,
    conjugate,
    coordinate,
    directional_derivative,
    dual_space,
    form,
    function,
    gateaux_derivative,
    gradient,
    integral,
    multiply,
    num,
    pull_back,
)
from uflx_spec_interpreter._src.meshes import cell_map
from uflx_spec_interpreter._src.passes.contracts import in_core_normal_form
from uflx_toy.runtime import runtime_array
from uflx_toy.spaces import space

V = space("V")
v, w = function(V, "v"), function(V, "w")
l = function(dual_space(V), "l")
cells = cell_map(D, 0)


def test_the_core_normal_form_allows_the_classes_of_its_table():
    for e in (
        num(1),
        coordinate(K),
        gradient(u, D),
        gradient(gradient(u, D), D),
        integral(num(1), K),
        pull_back(u, cells),
        form(multiply(v, w), (v,)),
        action(l, 0, conjugate(w)),
        runtime_array("q", (2,)),
    ):
        assert in_core_normal_form(e), e


def test_the_core_normal_form_rejects_what_its_table_leaves_out():
    for e in (
        gradient(coordinate(K), K),
        gradient(multiply(u, u), D),
        integral(u, D),
        pull_back(multiply(u, u), cells),
        # A cell of the dimension of the domain of the gradient, so the pass rewrites it.
        pull_back(gradient(u, D), cells),
        directional_derivative(multiply(v, v), v, v, w),
        gateaux_derivative(multiply(v, v), v, v, w),
        action(form(multiply(v, w), (v,)), 0, w),
        add(num(1), integral(u, D)),
    ):
        assert not in_core_normal_form(e), e
