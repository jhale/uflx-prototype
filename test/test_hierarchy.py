import pytest
from test_integrals import D, c

from uflx_oracle import evaluate
from uflx_spec_interpreter import cell_domain, conjugate_space, dual_space
from uflx_spec_interpreter._src import nodes
from uflx_spec_interpreter._src.nodes import (
    AbstractExpression,
    Base,
    Conjugate,
    ConjugateSpace,
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
from uflx_toy.cells import reference_simplex
from uflx_toy.nodes import IndexTable
from uflx_toy.spaces import space

EXPRESSIONS = (
    "Integer Float ImaginaryUnit Zero Sum Product Division Power Absolute Conjugate RealPart "
    "ImaginaryPart SquareRoot Logarithm Tensor Indexed ComponentTensor IndexSum Inverse "
    "Coordinate Gradient DirectionalDerivative Integral PullBack Normal Function Form Action "
    "GateauxDerivative"
).split()
STATIC_TYPE = {"value_shape", "free_indices", "argument_spaces", "support"}
OTHERS = {
    nodes: [
        "Index",
        "ConjugateSpace",
        "DualSpace",
        "GatheredIndex",
        "RD",
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


def test_a_curried_form_has_the_argument_spaces_of_its_expression_first():
    V, W = space("V"), space("W")
    u, v, w = Function(V, "u"), Function(V, "v"), Function(W, "w")
    inner = Form(Product(u, v), (v,))
    assert inner.argument_spaces == (V,)
    assert Form(Product(inner, w), (Conjugate(w),)).argument_spaces == (V, ConjugateSpace(W))
    assert Conjugate(inner).argument_spaces == (ConjugateSpace(V),)
    assert RealPart(u).argument_spaces == ()


def test_conjugating_an_argument_space_twice_gives_it_back():
    V, R = space("V"), space("R", scalar_type="real")
    assert conjugate_space(R) is R
    assert Conjugate(Conjugate(Form(Function(V, "v"), (Function(V, "v"),)))).argument_spaces == (V,)


def test_a_dual_space_has_the_conjugate_argument_space():
    V, Q = space("V"), space("Q")
    l = Function(dual_space(V), "l")
    assert l.argument_spaces == (ConjugateSpace(V),) and l.value_shape == ()
    assert dual_space((V, Q)).argument_spaces == ((ConjugateSpace(V), ConjugateSpace(Q)),)
    assert dual_space(dual_space(V)).argument_spaces == (ConjugateSpace(dual_space(V)),)


def test_every_expression_class_states_its_static_type():
    # The abstract class has no defaults, so each concrete class decides its own.
    for cls in concrete_classes():
        if issubclass(cls, AbstractExpression):
            own = set(vars(cls)) | set(cls.argument_names)
            assert STATIC_TYPE <= own, cls.__name__
    assert AbstractExpression.__abstractmethods__ >= STATIC_TYPE


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
    with pytest.raises(TypeError):
        nodes.AbstractCoordinateDomain()
    R = nodes.RD(2)
    assert R.dimension == 2 and R.ambient_domain is R and R.ambient_coordinate_domain is R
    assert isinstance(R, nodes.AbstractCoordinateDomain) and R != nodes.RD(3)


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


def test_a_reference_cell_is_a_coordinate_domain_and_a_cell_type():
    K = reference_simplex(2)
    assert isinstance(K, nodes.AbstractCoordinateDomain) and isinstance(K, nodes.AbstractCellType)
    assert K.dimension == 2 and K.num_vertices == 3 and K.vertices.value_shape == (3, 2)
    assert reference_simplex(0).vertices.value_shape == (1, 0)
    assert K != nodes.RD(2) and reference_simplex(0) != nodes.RD(0)
