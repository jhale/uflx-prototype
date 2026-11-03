from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Literal, final

from ..tuples import members
from .base import AbstractExpression, Base
from .scalar_types import ScalarType
from .spaces import DualSpace

if TYPE_CHECKING:
    from .domains import AbstractDomain, CellMap, DomainMap, Mesh, SideMap
    from .forms import Function


@final
class DirectionalDerivative(AbstractExpression):
    value_fields = ("expression", "point", "direction")
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

    def _validate(self):
        from ..validation import admissible, require, tuple_of, typed
        from .forms import Function

        typed(self.expression, AbstractExpression, "derivative requires an expression")
        xs = members(self.variable)
        tuple_of(xs, Function) if type(xs) is tuple else None
        for x in xs:
            typed(x, Function, "derivative variable must be a function")
            require(
                not x.argument_spaces or isinstance(x.space, DualSpace),
                "derivative variable is not bindable",
            )
        require(len(set(xs)) == len(xs), "derivative variables must be distinct")
        require(
            type(self.variable) is not tuple or len(xs) >= 2,
            "tuple derivative needs at least two variables",
        )
        for values in (self.point, self.direction):
            require(
                isinstance(values, tuple) == isinstance(self.variable, tuple)
                and len(members(values)) == len(xs),
                "derivative tuple structure differs",
            )
            for x, value in zip(xs, members(values), strict=True):
                admissible(x.space, value)


@final
class Coordinate(AbstractExpression):
    value_fields = ()
    is_terminal = True
    identifier = "uflx.Coordinate"
    domain: AbstractDomain

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

    def _validate(self):
        _validate_calculus(self)


@final
class Gradient(AbstractExpression):
    value_fields = ("expression",)
    identifier = "uflx.Gradient"
    commuting = ("expression",)
    expression: AbstractExpression
    domain: AbstractDomain

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

    def _validate(self):
        _validate_calculus(self)


@final
class Divergence(AbstractExpression):
    value_fields = ("expression",)
    identifier = "uflx.Divergence"
    commuting = ("expression",)
    # The trace of the gradient over the last axis of the value shape and the axis of the gradient.
    expression: AbstractExpression
    domain: AbstractDomain

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

    def _validate(self):
        _validate_calculus(self)


@final
class Curl(AbstractExpression):
    value_fields = ("expression",)
    identifier = "uflx.Curl"
    commuting = ("expression",)
    # The curl of a vector field, contracted with the permutation tensor of rank n.
    expression: AbstractExpression
    domain: AbstractDomain

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

    def _validate(self):
        _validate_calculus(self)


@final
class PullBack(AbstractExpression):
    value_fields = ("expression",)
    identifier = "uflx.PullBack"
    commuting = ("expression",)
    expression: AbstractExpression
    map: DomainMap | SideMap | CellMap

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

    def _validate(self):
        from ..validation import merge_indices, require, typed
        from .domains import CellMap, DomainMap, SideMap

        typed(self.expression, AbstractExpression, "pull-back requires an expression")
        typed(self.map, (DomainMap, CellMap, SideMap), "pull-back requires a map")
        require(
            self.expression.domain in (None, self.map.ambient_domain),
            "pull-back target differs from expression domain",
        )
        merge_indices(self.expression.free_indices, self.map.free_indices)


Variance = Literal["covariant", "contravariant"]


@final
class TransformationLaw(Base):
    identifier = "uflx.TransformationLaw"
    # The transformation law of a tensor density: a variance for each of the last axes, the
    # others unchanged, and a weight.
    variances: tuple[Variance, ...]
    weight: int

    def _validate(self):
        from ..validation import require, tuple_of

        tuple_of(self.variances, str)
        require(
            all(v in ("covariant", "contravariant") for v in self.variances), "unknown variance"
        )
        if type(self.weight) is not int:
            raise TypeError("weight must be an integer")


@final
class PushForward(AbstractExpression):
    value_fields = ("expression",)
    identifier = "uflx.PushForward"
    commuting = ("expression",)
    # The field on a mesh of a function on its topology, carried by a transformation law.
    expression: AbstractExpression
    mesh: Mesh
    law: TransformationLaw

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

    def _validate(self):
        from ..validation import require, typed
        from .domains import Mesh

        typed(self.expression, AbstractExpression, "push-forward requires an expression")
        typed(self.mesh, Mesh, "push-forward requires a mesh")
        typed(self.law, TransformationLaw, "push-forward requires an explicit law")
        require(
            self.expression.domain in (None, self.mesh.topology),
            "push-forward source differs from the topology",
        )
        p = len(self.law.variances)
        require(
            len(self.expression.value_shape) >= p
            and (not p or self.expression.value_shape[-p:] == (self.mesh.dimension,) * p),
            "transformed axes have the wrong extent",
        )


@final
class Integral(AbstractExpression):
    value_fields = ("expression",)
    identifier = "uflx.Integral"
    commuting = ("expression",)
    expression: AbstractExpression
    integration_domain: AbstractDomain

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

    def _validate(self):
        _validate_calculus(self)


@final
class Normal(AbstractExpression):
    value_fields = ()
    is_terminal = True
    identifier = "uflx.Normal"
    side_map: SideMap

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

    def _validate(self):
        from ..validation import require, typed
        from .domains import Mesh, SideMap

        typed(self.side_map, SideMap, "normal requires a side map")
        require(
            isinstance(self.side_map.ambient_domain, Mesh)
            and self.side_map.parameter_domain.dimension
            == self.side_map.ambient_domain.dimension - 1,
            "normal requires a codimension-one mesh side",
        )


def _validate_calculus(e):
    from ..validation import require, typed
    from .domains import AbstractDomain, Mesh

    P = e.integration_domain if isinstance(e, Integral) else e.domain
    typed(P, AbstractDomain, "calculus requires a domain")
    if not isinstance(e, Coordinate):
        f = e.expression
        typed(f, AbstractExpression, "calculus requires an expression")
        domains = (
            (None, P, P.ambient_domain)
            if isinstance(e, Integral) and isinstance(P, Mesh)
            else (None, P)
        )
        require(f.domain in domains, "expression and domain are not a valid pair")
        dim = P.ambient_coordinate_domain.dimension
        if isinstance(e, Divergence):
            require(
                bool(f.value_shape) and f.value_shape[-1] == dim,
                "divergence requires a last axis in the ambient frame",
            )
        if isinstance(e, Curl):
            require(
                dim in (2, 3) and f.value_shape == (dim,),
                "curl requires a vector in dimension two or three",
            )
