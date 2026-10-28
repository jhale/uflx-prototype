from test_facets import table
from test_integrals import D, c

from uflx_spec_interpreter import nodes
from uflx_spec_interpreter.domains import cell_domain
from uflx_spec_interpreter.nodes import (
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
    Space,
)
from uflx_spec_interpreter.passes import pullbacks
from uflx_spec_interpreter.static import free

EXPRESSIONS = (
    "Num Zero Sum Product Division Power Absolute Conjugate RealPart ImaginaryPart SquareRoot "
    "Logarithm Tensor Indexed ComponentTensor IndexSum Inverse Coordinate Gradient "
    "DirectionalDerivative Integral PullBack Normal Function Form Action GateauxDerivative"
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
    g = GatheredIndex(table([1, 0]), (c,))
    out = pullbacks.pb(Indexed(Coordinate(D), (g,)), cell_domain(D, 0))
    assert isinstance(out, Indexed) and out.indices == (g,) and out.indices[0] is g
    assert not isinstance(out.expression, PullBack) and not isinstance(out.expression, Function)


def test_a_curried_form_counts_the_arguments_of_its_expression():
    V = Space("V")
    u, v, w = (Function(V, n) for n in "uvw")
    inner = Form(Product(u, v), (v,))
    assert inner.arity == 1
    assert Form(Product(inner, w), (w,)).arity == 2
    assert RealPart(inner).arity == 1


def test_most_classes_use_the_defaults():
    # A class states only the parts of its static type that differ from the defaults.
    for name in (
        "SquareRoot",
        "Logarithm",
        "Absolute",
        "RealPart",
        "ImaginaryPart",
        "Division",
        "Power",
    ):
        own = vars(getattr(nodes, name))
        assert not {"value_shape", "free_indices", "arity", "support"} & set(own), name
