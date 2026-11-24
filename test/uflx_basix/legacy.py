# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT

# The classic FEniCS layer of EXAMPLES.md, for functions on a topology. A function of the layer
# is the push-forward of a function on the topology, so a user script has one name for each.
import basix

from uflx._src.nodes import AbstractExpression, Base, Function
from uflx.arithmetic import conjugate
from uflx.calculus import divergence, gradient, integral
from uflx.forms import form, function
from uflx.geometry import pull_back
from uflx.nodes import ScalarType
from uflx.tensors import index, index_sum, tensor
from uflx_basix.spaces import local_space

from ..uflx.sides import outward_conormal
from ..uflx.spaces import space as field_space
from .meshes import push_forward, side_domain, side_map


class Mesh:
    def __init__(self, domain, boundary):
        self.domain, self.boundary = domain, boundary


class FunctionSpace:
    # A space on the topology of the mesh, with the mesh that its functions are pushed to.
    def __init__(self, mesh, family, degree, shape=()):
        # The layer offers the Lagrange elements only, which the family names.
        self.mesh, self.shape = mesh, shape
        topology = mesh.domain.topology
        elements = tuple(
            basix.create_element(basix.ElementFamily.P, K.cell_type, degree)
            for K in topology.cell_types
        )
        self.space = local_space(topology, elements)

    def function(self, label):
        # The toy has no blocked spaces, so a vector function is a tensor of its components.
        D = self.mesh.domain
        if self.shape == ():
            return push_forward(function(self.space, label), D)
        return tensor(
            tuple(push_forward(function(self.space, (label, k)), D) for k in range(self.shape[0]))
        )


def VectorFunctionSpace(mesh, family, degree):
    return FunctionSpace(mesh, family, degree, (mesh.domain.ambient_domain.dimension,))


def TestFunction(V):
    return V.function(("argument", 0))


def TrialFunction(V):
    return V.function(("argument", 1))


def Coefficient(V, name):
    return V.function(name)


def Constant(name, shape=()):
    return function(field_space(name, shape, None, ScalarType.real), name)


def FacetNormal(mesh):
    # A placeholder on the mesh, which restrict replaces by the normal of the facets.
    n = mesh.domain.ambient_domain.dimension
    return function(field_space("facet normal", (n,), mesh.domain, ScalarType.real), "facet normal")


def is_facet_normal(obj):
    return isinstance(obj, Function) and obj.label == "facet normal"


def grad(f):
    return gradient(f, f.domain)


def div(w):
    return divergence(w, w.domain)


def inner(a, b):
    if a.value_shape == ():
        return a * conjugate(b)
    i = index("inner")
    return index_sum(a[i] * conjugate(b[i]), (i,))


def contains(obj, predicate):
    if predicate(obj):
        return True
    if isinstance(obj, tuple):
        return any(contains(item, predicate) for item in obj)
    return isinstance(obj, Base) and any(
        contains(item, predicate) for item in obj.construction_arguments
    )


def arguments(expression):
    # The functions labelled ("argument", n), by n, which the walk finds inside the push-forwards.
    found = {}

    def visit(obj):
        if isinstance(obj, Function) and isinstance(obj.label, tuple):
            if obj.label[0] == "argument":
                found[obj.label[1]] = obj
        if isinstance(obj, tuple):
            for item in obj:
                visit(item)
        elif isinstance(obj, Base):
            for item in obj.construction_arguments:
                visit(item)

    visit(expression)
    return tuple(found[n] for n in sorted(found))


def restrict(obj, E):
    if is_facet_normal(obj):
        return outward_conormal(side_map(obj.domain, E, "inside"))
    if isinstance(obj, tuple):
        return tuple(restrict(item, E) for item in obj)
    if not isinstance(obj, AbstractExpression) or obj.domain is None:
        return obj
    if not contains(obj, is_facet_normal):
        return pull_back(obj, side_map(obj.domain, E, "inside"))
    return obj.__class__(*restrict(obj.construction_arguments, E))


class Measure:
    def __init__(self, integral_type, domain=None, subdomain_id="everywhere", subdomain_data=None):
        self.integral_type = integral_type
        self.domain = domain
        self.subdomain_id = subdomain_id
        self.subdomain_data = subdomain_data

    def __call__(self, subdomain_id="everywhere", domain=None, subdomain_data=None):
        return Measure(
            self.integral_type,
            self.domain if domain is None else domain,
            subdomain_id,
            self.subdomain_data if subdomain_data is None else subdomain_data,
        )

    def __rmul__(self, integrand):
        mesh = self.domain
        if self.subdomain_id != "everywhere":
            E = side_domain(
                mesh.domain,
                self.subdomain_data[self.subdomain_id],
            )
        elif self.integral_type == "ds":
            E = side_domain(mesh.domain, mesh.boundary)
        else:
            E = integrand.domain if mesh is None else mesh.domain
        if E != integrand.domain:
            integrand = restrict(integrand, E)
        expression = integral(integrand, E)
        args = arguments(expression)
        if not args:
            return expression
        return form(expression, (conjugate(args[0]), *args[1:]))
