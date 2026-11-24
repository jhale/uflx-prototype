# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT

import pytest

from uflx._src import nodes
from uflx._src.algorithms.traversal import rebuild
from uflx._src.nodes import (
    AbstractExpression,
    Base,
    Conjugate,
    ConjugateSpace,
    Form,
    Function,
    Product,
    RealPart,
)
from uflx.forms import conjugate_space, dual_space
from uflx.nodes import ScalarType

from .spaces import space

EXPRESSIONS = (
    "Integer Float Complex Zero Sum Product Division Power Absolute Conjugate RealPart "
    "ImaginaryPart SquareRoot Logarithm Tensor Indexed ComponentTensor IndexSum Inverse "
    "Coordinate Gradient DirectionalDerivative Integral PullBack Conormal Function Form Apply "
).split()
STATIC_TYPE = {"value_shape", "free_indices", "argument_spaces", "domain", "scalar_type"}
OTHERS = [
    "Index",
    "ConjugateSpace",
    "DualSpace",
    "GatheredIndex",
    "EuclideanSpace",
    "Mesh",
    "DomainMap",
    "SideMap",
]


def test_the_classes_follow_the_hierarchy_of_the_specification():
    for name in EXPRESSIONS:
        assert issubclass(getattr(nodes, name), AbstractExpression), name
    for name in OTHERS:
        cls = getattr(nodes, name)
        assert issubclass(cls, Base) and not issubclass(cls, AbstractExpression), name


def test_a_curried_form_has_the_argument_spaces_of_its_expression_first():
    V, W = space("V", None), space("W", None)
    u, v, w = Function(V, "u"), Function(V, "v"), Function(W, "w")
    inner = Form(Product(u, v), (v,))
    assert inner.argument_spaces == (V,)
    assert Form(Product(inner, w), (Conjugate(w),)).argument_spaces == (V, ConjugateSpace(W))
    assert Conjugate(inner).argument_spaces == (ConjugateSpace(V),)
    assert RealPart(u).argument_spaces == ()


def test_conjugating_an_argument_space_twice_gives_it_back():
    V, R = space("V", None), space("R", None, scalar_type=ScalarType.real)
    assert conjugate_space(R) is R
    assert Conjugate(Conjugate(Form(Function(V, "v"), (Function(V, "v"),)))).argument_spaces == (V,)


def test_a_dual_space_has_the_conjugate_argument_space():
    V, Q = space("V", None), space("Q", None)
    l = Function(dual_space(V), "l")
    assert l.argument_spaces == (ConjugateSpace(V),) and l.value_shape == ()
    assert dual_space((V, Q)).argument_spaces == ((ConjugateSpace(V), ConjugateSpace(Q)),)
    assert dual_space(dual_space(V)) is V


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
    R = nodes.EuclideanSpace(2)
    assert R.dimension == 2 and R.ambient_domain is R and R.ambient_coordinate_domain is R
    assert isinstance(R, nodes.AbstractCoordinateDomain) and R != nodes.EuclideanSpace(3)


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
