from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast, final

from .arguments import _argument_members
from .base import AbstractExpression
from .interning import _cached
from .scalar_types import ScalarType
from .spaces import DualSpace

if TYPE_CHECKING:
    from .domains import AbstractDomain
    from .forms import Function
    from .indices import AbstractExtent, Index
    from .spaces import AbstractFunctionSpace
    from .topology import AbstractTopology


@final
class DirectionalDerivative(AbstractExpression):
    value_fields = ("expression", "point", "direction")
    identifier = "uflx.DirectionalDerivative"
    # The variable, the point and the direction are a function and two expressions, or tuples.
    expression: AbstractExpression
    variable: Function | tuple[Function, ...]
    point: AbstractExpression | tuple[AbstractExpression, ...]
    direction: AbstractExpression | tuple[AbstractExpression, ...]
    scope = ("expression", "variable")

    def bound(self) -> tuple[Function, ...]:
        return _argument_members(self.variable)

    @property
    @_cached
    def value_shape(self) -> tuple[int | AbstractExtent, ...]:
        return self.expression.value_shape

    @property
    @_cached
    def free_indices(self) -> tuple[tuple[Index, int | AbstractExtent], ...]:
        edges = (
            self.expression,
            *_argument_members(self.point),
            *_argument_members(self.direction),
        )
        from .static import _free_indices

        return _free_indices(*edges)

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
        from .validation import _admissible, _require

        xs = _argument_members(self.variable)
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
                and len(_argument_members(values)) == len(xs),
                "derivative tuple structure differs",
            )
            for x, value in zip(xs, _argument_members(values), strict=True):
                _admissible(x.space, value)


@final
class Coordinate(AbstractExpression):
    value_fields = ()
    is_terminal = True
    identifier = "uflx.Coordinate"
    __match_args__ = ("domain",)

    def __init__(self, domain: AbstractDomain, /) -> None:
        # Base.__new__ stores and validates the construction arguments.
        pass

    @property
    def domain(self) -> AbstractDomain:
        return cast("AbstractDomain", self._argument("domain"))

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
    __match_args__ = ("expression", "domain")

    def __init__(self, expression: AbstractExpression, domain: AbstractDomain, /) -> None:
        # Base.__new__ stores and validates the construction arguments.
        pass

    expression: AbstractExpression

    @property
    def domain(self) -> AbstractDomain:
        return cast("AbstractDomain", self._argument("domain"))

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
    # The trace of the gradient over the last axis of the value shape and the axis of the gradient.
    __match_args__ = ("expression", "domain")

    def __init__(self, expression: AbstractExpression, domain: AbstractDomain, /) -> None:
        # Base.__new__ stores and validates the construction arguments.
        pass

    expression: AbstractExpression

    @property
    def domain(self) -> AbstractDomain:
        return cast("AbstractDomain", self._argument("domain"))

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
    # The curl of a vector field, contracted with the permutation tensor of rank n.
    __match_args__ = ("expression", "domain")

    def __init__(self, expression: AbstractExpression, domain: AbstractDomain, /) -> None:
        # Base.__new__ stores and validates the construction arguments.
        pass

    expression: AbstractExpression

    @property
    def domain(self) -> AbstractDomain:
        return cast("AbstractDomain", self._argument("domain"))

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
class Integral(AbstractExpression):
    value_fields = ("expression",)
    identifier = "uflx.Integral"
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


def _validate_calculus(e: Coordinate | Gradient | Divergence | Curl | Integral) -> None:
    from .meshes import Mesh
    from .validation import _require

    P = e.integration_domain if isinstance(e, Integral) else e.domain
    if isinstance(e, Coordinate):
        return
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
