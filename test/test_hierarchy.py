from test_facets import table
from test_integrals import D, c

from uflx_spec_interpreter import domains, nodes
from uflx_spec_interpreter.domains import Gathered, cell_domain
from uflx_spec_interpreter.nodes import AbstractExpression, Base, Coordinate, Fn, Indexed, PullBack
from uflx_spec_interpreter.passes import pullbacks
from uflx_spec_interpreter.static import free

EXPRESSIONS = (
    "Num Zero Sum Prod Div Power Abs Conj Re Im Sqrt Log Tensor Indexed CT ISum Inv "
    "Coordinate Gradient D Integral PullBack Normal Fn Form Action Gateaux"
).split()
OTHERS = {
    nodes: ["Index", "Space"],
    domains: ["Gathered", "CoordDomain", "Decomposed", "DomainMap", "CellMap", "SideMap"],
}


def test_the_classes_follow_the_hierarchy_of_the_specification():
    for name in EXPRESSIONS:
        assert issubclass(getattr(nodes, name), AbstractExpression), name
    for module, names in OTHERS.items():
        for name in names:
            cls = getattr(module, name)
            assert issubclass(cls, Base) and not issubclass(cls, AbstractExpression), name


def test_the_static_type_is_computed_once():
    e = Indexed(Coordinate(D), (0,))
    assert e.value_shape is e.value_shape
    assert e.free_indices is e.free_indices
    assert free(e) == free(e) and free(e) is not free(e)


def test_a_pull_back_leaves_a_gathered_entry_unchanged():
    g = Gathered(table([1, 0]), (c,))
    out = pullbacks.pb(Indexed(Coordinate(D), (g,)), cell_domain(D, 0))
    assert isinstance(out, Indexed) and out.idx == (g,) and out.idx[0] is g
    assert not isinstance(out.e, PullBack) and not isinstance(out.e, Fn)
