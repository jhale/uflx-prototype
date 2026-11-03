from __future__ import annotations

from typing import TYPE_CHECKING, Literal, final

from ..interning import _cached
from ..tuples import _members
from .base import AbstractExpression, Base
from .scalar_types import ScalarType
from .spaces import DualSpace

if TYPE_CHECKING:
    from ..nodes.indices import AbstractExtent, Index
    from ..nodes.spaces import AbstractFunctionSpace
    from ..nodes.topology import AbstractTopology
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

    def bound(self) -> tuple[Function, ...]:
        return _members(self.variable)

    @property
    @_cached
    def value_shape(self) -> tuple[int | AbstractExtent, ...]:
        return self.expression.value_shape

    @property
    @_cached
    def free_indices(self) -> tuple[tuple[Index, int | AbstractExtent], ...]:
        edges = (self.expression, *_members(self.point), *_members(self.direction))
        return tuple(dict(p for x in edges for p in x.free_indices).items())

    @property
    @_cached
    def argument_spaces(
        self,
    ) -> tuple[AbstractFunctionSpace | tuple[AbstractFunctionSpace, ...], ...]:
        return self.expression.argument_spaces

    @property
    @_cached
    def scalar_type(self) -> Literal["real", "complex"]:
        return self.expression.scalar_type

    @property
    @_cached
    def domain(self) -> AbstractDomain | AbstractTopology | None:
        return self.expression.domain

    def _validate(self) -> None:
        from ..validation import _admissible, _require

        xs = _members(self.variable)
        for x in xs:
            _require(
                not x.argument_spaces or isinstance(x.space, DualSpace),
                "derivative variable is not bindable",
            )
        _require(len(set(xs)) == len(xs), "derivative variables must be distinct")
        _require(
            type(self.variable) is not tuple or len(xs) >= 2,
            "tuple derivative needs at least two variables",
        )
        for values in (self.point, self.direction):
            _require(
                isinstance(values, tuple) == isinstance(self.variable, tuple)
                and len(_members(values)) == len(xs),
                "derivative tuple structure differs",
            )
            for x, value in zip(xs, _members(values), strict=True):
                _admissible(x.space, value)


@final
class Coordinate(AbstractExpression):
    value_fields = ()
    is_terminal = True
    identifier = "uflx.Coordinate"
    domain: AbstractDomain  # pyright: ignore[reportIncompatibleMethodOverride]

    @property
    @_cached
    def value_shape(self) -> tuple[int | AbstractExtent, ...]:
        return (self.domain.ambient_coordinate_domain.dimension,)

    @property
    @_cached
    def scalar_type(self) -> Literal["real", "complex"]:
        return ScalarType.real

    @property
    @_cached
    def free_indices(self) -> tuple[tuple[Index, int | AbstractExtent], ...]:
        return ()

    @property
    @_cached
    def argument_spaces(
        self,
    ) -> tuple[AbstractFunctionSpace | tuple[AbstractFunctionSpace, ...], ...]:
        return ()

    def _validate(self) -> None:
        _validate_calculus(self)


@final
class Gradient(AbstractExpression):
    value_fields = ("expression",)
    identifier = "uflx.Gradient"
    commuting = ("expression",)
    expression: AbstractExpression
    domain: AbstractDomain  # pyright: ignore[reportIncompatibleMethodOverride]

    @property
    @_cached
    def value_shape(self) -> tuple[int | AbstractExtent, ...]:
        return (*self.expression.value_shape, self.domain.ambient_coordinate_domain.dimension)

    @property
    @_cached
    def scalar_type(self) -> Literal["real", "complex"]:
        return self.joined_scalar_type(self.expression)

    @property
    @_cached
    def free_indices(self) -> tuple[tuple[Index, int | AbstractExtent], ...]:
        return self.expression.free_indices

    @property
    @_cached
    def argument_spaces(
        self,
    ) -> tuple[AbstractFunctionSpace | tuple[AbstractFunctionSpace, ...], ...]:
        return self.expression.argument_spaces

    def _validate(self) -> None:
        _validate_calculus(self)


@final
class Divergence(AbstractExpression):
    value_fields = ("expression",)
    identifier = "uflx.Divergence"
    commuting = ("expression",)
    # The trace of the gradient over the last axis of the value shape and the axis of the gradient.
    expression: AbstractExpression
    domain: AbstractDomain  # pyright: ignore[reportIncompatibleMethodOverride]

    @property
    @_cached
    def value_shape(self) -> tuple[int | AbstractExtent, ...]:
        return self.expression.value_shape[:-1]

    @property
    @_cached
    def scalar_type(self) -> Literal["real", "complex"]:
        return self.joined_scalar_type(self.expression)

    @property
    @_cached
    def free_indices(self) -> tuple[tuple[Index, int | AbstractExtent], ...]:
        return self.expression.free_indices

    @property
    @_cached
    def argument_spaces(
        self,
    ) -> tuple[AbstractFunctionSpace | tuple[AbstractFunctionSpace, ...], ...]:
        return self.expression.argument_spaces

    def _validate(self) -> None:
        _validate_calculus(self)


@final
class Curl(AbstractExpression):
    value_fields = ("expression",)
    identifier = "uflx.Curl"
    commuting = ("expression",)
    # The curl of a vector field, contracted with the permutation tensor of rank n.
    expression: AbstractExpression
    domain: AbstractDomain  # pyright: ignore[reportIncompatibleMethodOverride]

    @property
    @_cached
    def value_shape(self) -> tuple[int | AbstractExtent, ...]:
        # The axes of the permutation tensor that the curl leaves, n - 2 of them.
        n = self.domain.ambient_coordinate_domain.dimension
        return (n,) * (n - 2)

    @property
    @_cached
    def scalar_type(self) -> Literal["real", "complex"]:
        return self.joined_scalar_type(self.expression)

    @property
    @_cached
    def free_indices(self) -> tuple[tuple[Index, int | AbstractExtent], ...]:
        return self.expression.free_indices

    @property
    @_cached
    def argument_spaces(
        self,
    ) -> tuple[AbstractFunctionSpace | tuple[AbstractFunctionSpace, ...], ...]:
        return self.expression.argument_spaces

    def _validate(self) -> None:
        _validate_calculus(self)


@final
class PullBack(AbstractExpression):
    value_fields = ("expression",)
    identifier = "uflx.PullBack"
    commuting = ("expression",)
    expression: AbstractExpression
    map: DomainMap | SideMap | CellMap

    @property
    @_cached
    def value_shape(self) -> tuple[int | AbstractExtent, ...]:
        return self.expression.value_shape

    @property
    @_cached
    def free_indices(self) -> tuple[tuple[Index, int | AbstractExtent], ...]:
        return tuple((dict(self.expression.free_indices) | dict(self.map.free_indices)).items())

    @property
    @_cached
    def scalar_type(self) -> Literal["real", "complex"]:
        return self.joined_scalar_type(self.expression)

    @property
    @_cached
    def domain(self) -> AbstractDomain | AbstractTopology | None:
        return self.map.parameter_domain

    @property
    @_cached
    def argument_spaces(
        self,
    ) -> tuple[AbstractFunctionSpace | tuple[AbstractFunctionSpace, ...], ...]:
        return self.expression.argument_spaces

    def _validate(self) -> None:
        from ..validation import _merge_indices, _require

        _require(
            self.expression.domain in (None, self.map.ambient_domain),
            "pull-back target differs from expression domain",
        )
        _merge_indices(self.expression.free_indices, self.map.free_indices)


_Variance = Literal["covariant", "contravariant"]


@final
class TransformationLaw(Base):
    identifier = "uflx.TransformationLaw"
    # The transformation law of a tensor density: a variance for each of the last axes, the
    # others unchanged, and a weight.
    variances: tuple[_Variance, ...]
    weight: int

    def _validate(self) -> None:
        from ..validation import _require

        _require(
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

    @property
    @_cached
    def value_shape(self) -> tuple[int | AbstractExtent, ...]:
        # A covariant or a contravariant axis has the extent k on the reference cell and n on D.
        shape, p = self.expression.value_shape, len(self.law.variances)
        n = self.mesh.ambient_domain.ambient_coordinate_domain.dimension
        return (*shape[: len(shape) - p], *(n for _ in range(p)))

    @property
    @_cached
    def free_indices(self) -> tuple[tuple[Index, int | AbstractExtent], ...]:
        return self.expression.free_indices

    @property
    @_cached
    def scalar_type(self) -> Literal["real", "complex"]:
        return self.joined_scalar_type(self.expression)

    @property
    @_cached
    def domain(self) -> AbstractDomain | AbstractTopology | None:
        return self.mesh

    @property
    @_cached
    def argument_spaces(
        self,
    ) -> tuple[AbstractFunctionSpace | tuple[AbstractFunctionSpace, ...], ...]:
        return self.expression.argument_spaces

    def _validate(self) -> None:
        from ..validation import _require

        _require(
            self.expression.domain in (None, self.mesh.topology),
            "push-forward source differs from the topology",
        )
        p = len(self.law.variances)
        _require(
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

    @property
    @_cached
    def value_shape(self) -> tuple[int | AbstractExtent, ...]:
        return self.expression.value_shape

    @property
    @_cached
    def free_indices(self) -> tuple[tuple[Index, int | AbstractExtent], ...]:
        # Mesh integration binds its own cell index, never an integrand index.
        return self.expression.free_indices

    @property
    @_cached
    def scalar_type(self) -> Literal["real", "complex"]:
        return self.joined_scalar_type(self.expression)

    @property
    @_cached
    def domain(self) -> AbstractDomain | AbstractTopology | None:
        return None

    @property
    @_cached
    def argument_spaces(
        self,
    ) -> tuple[AbstractFunctionSpace | tuple[AbstractFunctionSpace, ...], ...]:
        return self.expression.argument_spaces

    def _validate(self) -> None:
        _validate_calculus(self)


@final
class Normal(AbstractExpression):
    value_fields = ()
    is_terminal = True
    identifier = "uflx.Normal"
    side_map: SideMap

    @property
    @_cached
    def value_shape(self) -> tuple[int | AbstractExtent, ...]:
        from .domains import Mesh

        domain = self.side_map.ambient_domain
        if not isinstance(domain, Mesh):
            raise TypeError("normal requires a mesh side")
        return (domain.ambient_coordinate_domain.dimension,)

    @property
    @_cached
    def scalar_type(self) -> Literal["real", "complex"]:
        return ScalarType.real

    @property
    @_cached
    def domain(self) -> AbstractDomain | AbstractTopology | None:
        return self.side_map.parameter_domain

    @property
    @_cached
    def free_indices(self) -> tuple[tuple[Index, int | AbstractExtent], ...]:
        return ()

    @property
    @_cached
    def argument_spaces(
        self,
    ) -> tuple[AbstractFunctionSpace | tuple[AbstractFunctionSpace, ...], ...]:
        return ()

    def _validate(self) -> None:
        from ..validation import _require
        from .domains import Mesh

        _require(
            isinstance(self.side_map.ambient_domain, Mesh)
            and self.side_map.parameter_domain.dimension
            == self.side_map.ambient_domain.dimension - 1,
            "normal requires a codimension-one mesh side",
        )


def _validate_calculus(e: Coordinate | Gradient | Divergence | Curl | Integral) -> None:
    from ..validation import _require
    from .domains import Mesh

    P = e.integration_domain if isinstance(e, Integral) else e.domain
    if not isinstance(e, Coordinate):
        f = e.expression
        domains = (
            (None, P, P.ambient_domain)
            if isinstance(e, Integral) and isinstance(P, Mesh)
            else (None, P)
        )
        _require(f.domain in domains, "expression and domain are not a valid pair")
        dim = P.ambient_coordinate_domain.dimension
        if isinstance(e, Divergence):
            _require(
                bool(f.value_shape) and f.value_shape[-1] == dim,
                "divergence requires a last axis in the ambient frame",
            )
        if isinstance(e, Curl):
            _require(
                dim in (2, 3) and f.value_shape == (dim,),
                "curl requires a vector in dimension two or three",
            )
