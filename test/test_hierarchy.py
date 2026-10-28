import pytest
from test_integrals import D, c

from uflx_oracle import evaluate
from uflx_spec_interpreter import cell_domain, space
from uflx_spec_interpreter._src import nodes
from uflx_spec_interpreter._src.nodes import (
    AbstractExpression,
    Base,
    Coordinate,
    Form,
    Function,
    GatheredIndex,
    Indexed,
    Product,
    PullBack,
    RealPart,
)
from uflx_spec_interpreter._src.passes import pullbacks
from uflx_spec_interpreter._src.static import free
from uflx_spec_interpreter._src.substitution import rebuild
from uflx_toy.nodes import IndexTable

EXPRESSIONS = (
    "Integer Float ImaginaryUnit Zero Sum Product Division Power Absolute Conjugate RealPart "
    "ImaginaryPart SquareRoot Logarithm Tensor Indexed ComponentTensor IndexSum Inverse "
    "Coordinate Gradient DirectionalDerivative Integral PullBack Normal Function Form Action "
    "GateauxDerivative"
).split()
OTHERS = {
    nodes: [
        "Index",
        "Space",
        "GatheredIndex",
        "CoordinateDomain",
        "DecomposedDomain",
        "DomainMap",
        "CellMap",
        "SideMap",
    ],
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
    g = GatheredIndex(IndexTable((1, 0), 2), (c,))
    out = pullbacks.pb(Indexed(Coordinate(D), (g,)), cell_domain(D, 0))
    assert isinstance(out, Indexed) and out.indices == (g,) and out.indices[0] is g
    assert not isinstance(out.expression, PullBack) and not isinstance(out.expression, Function)


def test_a_curried_form_counts_the_arguments_of_its_expression():
    V = space("V")
    u, v, w = (Function(V, n) for n in "uvw")
    inner = Form(Product(u, v), (v,))
    assert inner.arity == 1
    assert Form(Product(inner, w), (w,)).arity == 2
    assert RealPart(inner).arity == 1


def test_every_expression_class_states_its_static_type():
    # The abstract class has no defaults, so each concrete class decides its own.
    for cls in concrete_classes():
        if issubclass(cls, AbstractExpression):
            own = set(vars(cls)) | set(cls.argument_names)
            assert {"value_shape", "free_indices", "arity", "support"} <= own, cls.__name__
    assert AbstractExpression.__abstractmethods__ >= {
        "value_shape",
        "free_indices",
        "arity",
        "support",
    }


def concrete_classes():
    return [
        cls for cls in vars(nodes).values() if isinstance(cls, type) and "identifier" in vars(cls)
    ]


def test_every_concrete_class_is_final():
    for cls in concrete_classes():
        assert getattr(cls, "__final__", False), cls.__name__
    with pytest.raises(TypeError):

        class Larger(nodes.Sum):
            pass


def test_a_domain_is_a_contract():
    with pytest.raises(TypeError):
        nodes.AbstractDomain()
    K = nodes.CoordinateDomain("K", 2)
    assert K.dimension == 2 and K.ambient_domain is K and isinstance(K, nodes.AbstractDomain)


def test_nodes_are_immutable_and_take_their_arguments():
    e = nodes.Integer(1)
    with pytest.raises(AttributeError):
        e.value = 2
    with pytest.raises(TypeError):
        nodes.Integer()
    with pytest.raises(TypeError):
        nodes.Integer(1, 2)
    with pytest.raises(TypeError):
        rebuild(e, other=3)
    assert rebuild(e, value=2) is nodes.Integer(2)
    assert nodes.Sum.argument_names == ("first", "second")
    assert nodes.Form.argument_names == ("expression", "arguments")


def test_a_gathered_index_reads_an_index_value_and_gathers_nest():
    i, j = nodes.Index("i"), nodes.Index("j")
    table = IndexTable(((0, 1), (1, 0)), 2)
    assert table.shape == (2, 2) and table.extent == 2
    assert not isinstance(table, AbstractExpression)
    with pytest.raises(TypeError):
        nodes.AbstractIndexValue()
    inner = nodes.GatheredIndex(IndexTable((1, 0), 2), (j,))
    g = nodes.GatheredIndex(table, (i, inner))
    assert g.extent == 2 and g.free_indices == ((i, 2), (j, 2))
    a = nodes.Indexed(nodes.Tensor((nodes.Float(10.0), nodes.Float(20.0))), (g,))
    assert a.free_indices == ((i, 2), (j, 2))
    # With i = 0 and j = 0, the inner table gives 1, the outer table gives table[0][1] = 1.
    assert evaluate(a, {}, {i: 0, j: 0})(()) == 20.0
    assert evaluate(a, {}, {i: 1, j: 0})(()) == 10.0
