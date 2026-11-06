from __future__ import annotations

from typing import TYPE_CHECKING, Literal, final

from .caching import cached
from .expressions import AbstractExpression
from .indices import Index
from .validation import require

if TYPE_CHECKING:
    from .domains import AbstractDomain
    from .indices import AbstractExtent, GatheredIndex
    from .spaces import AbstractFunctionSpace
    from .topology import AbstractTopology


@final
class BroadcastTo(AbstractExpression):
    """Repeat a value along leading and singleton axes."""

    identifier = "uflx.BroadcastTo"
    value_fields = ("value",)

    value: AbstractExpression
    _value_shape: tuple[int | AbstractExtent, ...]

    @property
    def value_shape(self) -> tuple[int | AbstractExtent, ...]:
        return self._value_shape

    @property
    @cached
    def free_indices(self) -> tuple[tuple[Index, int | AbstractExtent], ...]:
        return self.value.free_indices

    @property
    @cached
    def argument_spaces(
        self,
    ) -> tuple[AbstractFunctionSpace | tuple[AbstractFunctionSpace, ...], ...]:
        return self.value.argument_spaces

    @property
    @cached
    def scalar_type(self) -> Literal["real", "complex"]:
        return self.value.scalar_type

    @property
    @cached
    def domain(self) -> AbstractDomain | AbstractTopology | None:
        return self.value.domain

    def _validate(self) -> None:
        from .validation import extent, require

        for n in self.value_shape:
            extent(n)
        source = self.value.value_shape
        require(len(source) <= len(self.value_shape), "broadcast target has too few axes")
        aligned_axes = zip(reversed(source), reversed(self.value_shape))
        compatible = all(
            source_extent == target_extent or source_extent == 1
            for source_extent, target_extent in aligned_axes
        )
        require(compatible, "broadcast axes must match or have source extent one")


@final
class Tensor(AbstractExpression):
    matching_domains = True
    value_fields = ("entries",)
    identifier = "uflx.Tensor"
    entries: tuple[AbstractExpression, ...]

    @property
    @cached
    def value_shape(self) -> tuple[int | AbstractExtent, ...]:
        return (len(self.entries), *self.entries[0].value_shape)

    @property
    @cached
    def free_indices(self) -> tuple[tuple[Index, int | AbstractExtent], ...]:
        from .static import free_indices

        return free_indices(*self.entries)

    @property
    @cached
    def argument_spaces(
        self,
    ) -> tuple[AbstractFunctionSpace | tuple[AbstractFunctionSpace, ...], ...]:
        return self.entries[0].argument_spaces

    @property
    @cached
    def scalar_type(self) -> Literal["real", "complex"]:
        return self.joined_scalar_type(*self.entries)

    @property
    @cached
    def domain(self) -> AbstractDomain | AbstractTopology | None:
        return next((x.domain for x in reversed(self.entries) if x.domain), None)

    def _validate(self) -> None:
        from .validation import require

        require(bool(self.entries), "tensor entries must be nonempty")
        require(
            all(
                f.value_shape == self.entries[0].value_shape
                and f.argument_spaces == self.entries[0].argument_spaces
                for f in self.entries
            ),
            "tensor entry types differ",
        )


@final
class Indexed(AbstractExpression):
    index_fields = ("indices",)
    value_fields = ("expression",)
    identifier = "uflx.Indexed"
    expression: AbstractExpression
    indices: tuple[int | Index | GatheredIndex, ...]

    @property
    @cached
    def free_indices(self) -> tuple[tuple[Index, int | AbstractExtent], ...]:
        from .validation import entries, merge_indices

        return merge_indices(
            self.expression.free_indices, entries(self.expression.value_shape, self.indices)
        )

    @property
    @cached
    def value_shape(self) -> tuple[int | AbstractExtent, ...]:
        return ()

    @property
    @cached
    def argument_spaces(
        self,
    ) -> tuple[AbstractFunctionSpace | tuple[AbstractFunctionSpace, ...], ...]:
        return self.expression.argument_spaces

    @property
    @cached
    def scalar_type(self) -> Literal["real", "complex"]:
        return self.joined_scalar_type(self.expression)

    @property
    @cached
    def domain(self) -> AbstractDomain | AbstractTopology | None:
        return self.expression.domain

    def _validate(self) -> None:
        from .validation import entries, merge_indices, require

        require(bool(self.indices), "Indexed requires nonempty entries")
        merge_indices(
            self.expression.free_indices, entries(self.expression.value_shape, self.indices)
        )


@final
class ComponentTensor(AbstractExpression):
    value_fields = ("expression",)
    identifier = "uflx.ComponentTensor"
    expression: AbstractExpression
    indices: tuple[Index, ...]
    scope = ("expression", "indices")
    binding_fields = ("indices",)

    def bound(self) -> tuple[Index, ...]:
        return self.indices

    @property
    @cached
    def value_shape(self) -> tuple[int | AbstractExtent, ...]:
        free = dict(self.expression.free_indices)
        return tuple(free[i] for i in self.indices)

    @property
    @cached
    def free_indices(self) -> tuple[tuple[Index, int | AbstractExtent], ...]:
        return tuple((i, n) for i, n in self.expression.free_indices if i not in self.indices)

    @property
    @cached
    def argument_spaces(
        self,
    ) -> tuple[AbstractFunctionSpace | tuple[AbstractFunctionSpace, ...], ...]:
        return self.expression.argument_spaces

    @property
    @cached
    def scalar_type(self) -> Literal["real", "complex"]:
        return self.joined_scalar_type(self.expression)

    @property
    @cached
    def domain(self) -> AbstractDomain | AbstractTopology | None:
        return self.expression.domain

    def _validate(self) -> None:
        _validate_index_binding(self)


@final
class IndexSum(AbstractExpression):
    value_fields = ("expression",)
    identifier = "uflx.IndexSum"
    expression: AbstractExpression
    indices: tuple[Index, ...]
    scope = ("expression", "indices")
    binding_fields = ("indices",)

    def bound(self) -> tuple[Index, ...]:
        return self.indices

    @property
    @cached
    def free_indices(self) -> tuple[tuple[Index, int | AbstractExtent], ...]:
        return tuple((i, n) for i, n in self.expression.free_indices if i not in self.indices)

    @property
    @cached
    def value_shape(self) -> tuple[int | AbstractExtent, ...]:
        return ()

    @property
    @cached
    def argument_spaces(
        self,
    ) -> tuple[AbstractFunctionSpace | tuple[AbstractFunctionSpace, ...], ...]:
        return self.expression.argument_spaces

    @property
    @cached
    def scalar_type(self) -> Literal["real", "complex"]:
        return self.joined_scalar_type(self.expression)

    @property
    @cached
    def domain(self) -> AbstractDomain | AbstractTopology | None:
        return self.expression.domain

    def _validate(self) -> None:
        _validate_index_binding(self)


@final
class Inverse(AbstractExpression):
    value_fields = ("operand",)
    identifier = "uflx.Inverse"
    operand: AbstractExpression

    @property
    @cached
    def value_shape(self) -> tuple[int | AbstractExtent, ...]:
        return tuple(reversed(self.operand.value_shape))

    @property
    @cached
    def free_indices(self) -> tuple[tuple[Index, int | AbstractExtent], ...]:
        return self.operand.free_indices

    @property
    @cached
    def argument_spaces(
        self,
    ) -> tuple[AbstractFunctionSpace | tuple[AbstractFunctionSpace, ...], ...]:
        return ()

    @property
    @cached
    def scalar_type(self) -> Literal["real", "complex"]:
        return self.joined_scalar_type(self.operand)

    @property
    @cached
    def domain(self) -> AbstractDomain | AbstractTopology | None:
        return self.operand.domain

    def _validate(self) -> None:
        _matrix_operand(self.operand, rectangular=True)


@final
class Determinant(AbstractExpression):
    value_fields = ("operand",)
    identifier = "uflx.Determinant"
    operand: AbstractExpression

    @property
    @cached
    def value_shape(self) -> tuple[int | AbstractExtent, ...]:
        return ()

    @property
    @cached
    def free_indices(self) -> tuple[tuple[Index, int | AbstractExtent], ...]:
        return self.operand.free_indices

    @property
    @cached
    def argument_spaces(
        self,
    ) -> tuple[AbstractFunctionSpace | tuple[AbstractFunctionSpace, ...], ...]:
        return ()

    @property
    @cached
    def scalar_type(self) -> Literal["real", "complex"]:
        return self.joined_scalar_type(self.operand)

    @property
    @cached
    def domain(self) -> AbstractDomain | AbstractTopology | None:
        return self.operand.domain

    def _validate(self) -> None:
        _matrix_operand(self.operand)


def _validate_index_binding(e: ComponentTensor | IndexSum) -> None:
    from .validation import require

    require(
        bool(e.indices) and len(set(e.indices)) == len(e.indices),
        "bound indices must be nonempty and distinct",
    )
    require(e.expression.value_shape == (), "index binding requires a scalar")
    require(all(i in dict(e.expression.free_indices) for i in e.indices), "bound index is not free")


def _matrix_operand(f: AbstractExpression, rectangular: bool = False) -> None:
    require(not f.argument_spaces, "nonlinear operand has argument spaces")
    require(len(f.value_shape) == 2, "matrix operand must have rank two")
    a, b = f.value_shape
    require(
        a == b
        or (
            rectangular
            and isinstance(a, int)
            and isinstance(b, int)
            and b < a
            and f.scalar_type == "real"
        ),
        "matrix dimensions or scalar type are invalid",
    )
