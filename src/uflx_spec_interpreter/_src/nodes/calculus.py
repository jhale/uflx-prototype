from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Literal, final

from ..tuples import members
from .base import AbstractExpression, Base
from .scalar_types import ScalarType

if TYPE_CHECKING:
    from .domains import AbstractDomain, CellMap, DomainMap, Mesh, SideMap
    from .forms import Function


@final
class DirectionalDerivative(AbstractExpression):
    identifier = "uflx.DirectionalDerivative"
    commuting = ("expression",)
    # The variable, the point and the direction are a function and two expressions, or tuples.
    expression: AbstractExpression
    variable: Function | tuple[Function, ...]
    point: AbstractExpression | tuple[AbstractExpression, ...]
    direction: AbstractExpression | tuple[AbstractExpression, ...]
    scope = ("expression", "variable")

    def bound(self):
        return members(self.variable)

    @cached_property
    def value_shape(self):
        return self.expression.value_shape

    @cached_property
    def free_indices(self):
        edges = (self.expression, *members(self.point), *members(self.direction))
        return tuple(dict(p for x in edges for p in x.free_indices).items())

    @cached_property
    def argument_spaces(self):
        return self.expression.argument_spaces

    @cached_property
    def scalar_type(self):
        return self.expression.scalar_type

    @cached_property
    def domain(self):
        return self.expression.domain


@final
class Coordinate(AbstractExpression):
    identifier = "uflx.Coordinate"
    domain: AbstractDomain
    opaque = ("domain",)

    @cached_property
    def value_shape(self):
        return (self.domain.ambient_coordinate_domain.dimension,)

    @cached_property
    def scalar_type(self):
        return ScalarType.real

    @cached_property
    def free_indices(self):
        return ()

    @cached_property
    def argument_spaces(self):
        return ()


@final
class Gradient(AbstractExpression):
    identifier = "uflx.Gradient"
    commuting = ("expression",)
    expression: AbstractExpression
    domain: AbstractDomain
    opaque = ("domain",)

    @cached_property
    def value_shape(self):
        return (*self.expression.value_shape, self.domain.ambient_coordinate_domain.dimension)

    @cached_property
    def scalar_type(self):
        return self.joined_scalar_type(self.expression)

    @cached_property
    def free_indices(self):
        return self.expression.free_indices

    @cached_property
    def argument_spaces(self):
        return self.expression.argument_spaces


@final
class Divergence(AbstractExpression):
    identifier = "uflx.Divergence"
    commuting = ("expression",)
    # The trace of the gradient over the last axis of the value shape and the axis of the gradient.
    expression: AbstractExpression
    domain: AbstractDomain
    opaque = ("domain",)

    @cached_property
    def value_shape(self):
        return self.expression.value_shape[:-1]

    @cached_property
    def scalar_type(self):
        return self.joined_scalar_type(self.expression)

    @cached_property
    def free_indices(self):
        return self.expression.free_indices

    @cached_property
    def argument_spaces(self):
        return self.expression.argument_spaces


@final
class Curl(AbstractExpression):
    identifier = "uflx.Curl"
    commuting = ("expression",)
    # The curl of a vector field, contracted with the permutation tensor of rank n.
    expression: AbstractExpression
    domain: AbstractDomain
    opaque = ("domain",)

    @cached_property
    def value_shape(self):
        # The axes of the permutation tensor that the curl leaves, n - 2 of them.
        n = self.domain.ambient_coordinate_domain.dimension
        return (n,) * (n - 2)

    @cached_property
    def scalar_type(self):
        return self.joined_scalar_type(self.expression)

    @cached_property
    def free_indices(self):
        return self.expression.free_indices

    @cached_property
    def argument_spaces(self):
        return self.expression.argument_spaces


@final
class PullBack(AbstractExpression):
    identifier = "uflx.PullBack"
    commuting = ("expression",)
    expression: AbstractExpression
    map: DomainMap | SideMap | CellMap
    objects = ("map",)

    @cached_property
    def value_shape(self):
        return self.expression.value_shape

    @cached_property
    def free_indices(self):
        return tuple((dict(self.expression.free_indices) | dict(self.map.free_indices)).items())

    @cached_property
    def scalar_type(self):
        return self.joined_scalar_type(self.expression)

    @cached_property
    def domain(self):
        return self.map.parameter_domain

    @cached_property
    def argument_spaces(self):
        return self.expression.argument_spaces


Variance = Literal["covariant", "contravariant"]


@final
class TransformationLaw(Base):
    identifier = "uflx.TransformationLaw"
    # The transformation law of a tensor density: a variance for each of the last axes, the
    # others unchanged, and a weight.
    variances: tuple[Variance, ...]
    weight: int
    opaque = ("variances", "weight")


@final
class PushForward(AbstractExpression):
    identifier = "uflx.PushForward"
    commuting = ("expression",)
    # The field on a mesh of a function on its topology, carried by a transformation law.
    expression: AbstractExpression
    mesh: Mesh
    law: TransformationLaw
    objects = ("mesh",)

    @cached_property
    def value_shape(self):
        # A covariant or a contravariant axis has the extent k on the reference cell and n on D.
        shape, p = self.expression.value_shape, len(self.law.variances)
        n = self.mesh.ambient_domain.ambient_coordinate_domain.dimension
        return (*shape[: len(shape) - p], *(n for _ in range(p)))

    @cached_property
    def free_indices(self):
        return self.expression.free_indices

    @cached_property
    def scalar_type(self):
        return self.joined_scalar_type(self.expression)

    @cached_property
    def domain(self):
        return self.mesh

    @cached_property
    def argument_spaces(self):
        return self.expression.argument_spaces


@final
class Integral(AbstractExpression):
    identifier = "uflx.Integral"
    commuting = ("expression",)
    expression: AbstractExpression
    integration_domain: AbstractDomain
    opaque = ("integration_domain",)

    @cached_property
    def value_shape(self):
        return self.expression.value_shape

    @cached_property
    def free_indices(self):
        # Mesh integration binds its own cell index, never an integrand index.
        return self.expression.free_indices

    @cached_property
    def scalar_type(self):
        return self.joined_scalar_type(self.expression)

    @cached_property
    def domain(self):
        return None

    @cached_property
    def argument_spaces(self):
        return self.expression.argument_spaces


@final
class Normal(AbstractExpression):
    identifier = "uflx.Normal"
    side_map: SideMap
    objects = ("side_map",)

    @cached_property
    def value_shape(self):
        return (self.side_map.ambient_domain.ambient_coordinate_domain.dimension,)

    @cached_property
    def scalar_type(self):
        return ScalarType.real

    @cached_property
    def domain(self):
        return self.side_map.parameter_domain

    @cached_property
    def free_indices(self):
        return ()

    @cached_property
    def argument_spaces(self):
        return ()
