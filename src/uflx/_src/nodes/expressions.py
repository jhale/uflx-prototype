# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT
"""Expression static types, validation and Python operator syntax."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar, Generic, Never, cast

from typing_extensions import TypeVar, override

from ..typing import ExpressionScalarType
from .arguments import argument_leaves, argument_members, binding_name
from .base import Base, construction_field
from .scalar_types import ScalarType

if TYPE_CHECKING:
    from ..typing.nodes import ArgumentSpaces, BindingEntry, IndexEntry
    from .domains import AbstractDomain
    from .forms import Function
    from .indices import AbstractExtent, Index
    from .topology import AbstractTopology


class AbstractExpression(Base, ABC):
    # Additional linear operands declared by extensions; core rules live in rules/.
    commuting: ClassVar[tuple[str, ...]] = ()

    def __post_init__(self) -> None:
        from .structure import common_operand_domain, expression_operands
        from .validation import merge_indices

        super().__post_init__()
        if self.is_terminal and self.value_fields != ():
            raise ValueError("a terminal has no value operands")
        operands = expression_operands(self)
        _ = merge_indices(*(x.free_indices for x in operands))
        if self.scalar_type == ScalarType.boolean and self.argument_spaces != ():
            raise ValueError("Boolean expressions cannot take arguments")
        if self.requires_matching_domains:
            _ = common_operand_domain(self)

    # Whether the result depends only on the operands' values at the current point.
    is_pointwise: ClassVar[bool] = False
    requires_matching_domains: ClassVar[bool] = False

    # Every concrete class states its static type, so the abstract class has no defaults.
    @property
    @abstractmethod
    def value_shape(self) -> tuple[int | AbstractExtent, ...]: ...

    @property
    @abstractmethod
    def free_indices(self) -> tuple[tuple[Index, int | AbstractExtent], ...]: ...

    @property
    @abstractmethod
    def argument_spaces(self) -> ArgumentSpaces: ...

    @property
    @abstractmethod
    def domain(self) -> AbstractDomain | AbstractTopology | None: ...

    @property
    @abstractmethod
    def scalar_type(self) -> ExpressionScalarType: ...

    # The free functions import the nodes, so each overload imports its free function locally.
    def __add__(self, other: object) -> AbstractExpression:
        from ..arithmetic import add

        if not isinstance(other, AbstractExpression):
            return NotImplemented
        return add(self, other)

    def __sub__(self, other: object) -> AbstractExpression:
        from ..arithmetic import subtract

        if not isinstance(other, AbstractExpression):
            return NotImplemented
        return subtract(self, other)

    def __mul__(self, other: object) -> AbstractExpression:
        from ..arithmetic import multiply

        if not isinstance(other, AbstractExpression):
            return NotImplemented
        return multiply(self, other)

    def __truediv__(self, other: object) -> AbstractExpression:
        from ..arithmetic import divide

        if not isinstance(other, AbstractExpression):
            return NotImplemented
        return divide(self, other)

    def __pow__(self, other: object) -> AbstractExpression:
        from ..arithmetic import power

        if not isinstance(other, AbstractExpression):
            return NotImplemented
        return power(self, other)

    def __matmul__(self, other: object) -> AbstractExpression:
        from ..tensors import matrix_product

        if not isinstance(other, AbstractExpression):
            return NotImplemented
        return matrix_product(self, other)

    def __neg__(self) -> AbstractExpression:
        from ..arithmetic import negate

        return negate(self)

    def __abs__(self) -> AbstractExpression:
        from ..arithmetic import absolute

        return absolute(self)

    def __getitem__(self, entries: IndexEntry | tuple[IndexEntry, ...]) -> AbstractExpression:
        from ..tensors import component

        return component(self, argument_members(entries))

    def __iter__(self) -> Never:
        raise TypeError(f"{type(self).__name__} is not iterable")

    def __bool__(self) -> Never:
        raise TypeError("symbolic expressions have no Python truth value")


_Name = TypeVar("_Name", bound="Function | Index", default="Function | Index")


class BindingExpression(AbstractExpression, Generic[_Name], ABC):
    """Bind the declared functions or indices in the expression body."""

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if "binding_fields" in vars(cls) and "scope" not in vars(cls):
            cls.scope = ("expression", *cls.binding_fields)

    def binding_entries(self) -> tuple[BindingEntry, ...]:
        """Outer declarations in order, preserving grouped form entries."""
        entries: list[BindingEntry] = []
        for field in self.binding_fields:
            value = cast("BindingEntry | tuple[BindingEntry, ...]", construction_field(self, field))
            entries.extend(value if isinstance(value, tuple) else (value,))
        return tuple(entries)

    @override
    def bound(self) -> tuple[_Name, ...]:
        names = (
            binding_name(member)
            for entry in self.binding_entries()
            for member in argument_leaves(entry)
        )
        return cast("tuple[_Name, ...]", tuple(names))
