"""Expression static types, validation and Python operator syntax."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, ClassVar, Never

from .arguments import argument_members
from .base import Base
from .scalar_types import ExpressionScalarType, ScalarType

if TYPE_CHECKING:
    from .domains import AbstractDomain
    from .indices import AbstractExtent, GatheredIndex, Index
    from .spaces import AbstractFunctionSpace
    from .topology import AbstractTopology


class AbstractExpression(Base):
    # Additional linear operands declared by extensions; core rules live in rules/.
    commuting: ClassVar[tuple[str, ...]] = ()

    def __post_init__(self) -> None:
        from .structure import expression_operands
        from .validation import merge_indices, require

        super().__post_init__()
        require(not self.is_terminal or not self.value_fields, "a terminal has no value operands")
        operands = expression_operands(self)
        merge_indices(*(x.free_indices for x in operands))
        if self.scalar_type == ScalarType.boolean:
            require(not self.argument_spaces, "Boolean expressions cannot take arguments")
        if self.matching_domains:
            domains = [x.domain for x in operands if x.domain is not None]
            require(not domains or all(D == domains[0] for D in domains), "operand domains differ")

    matching_domains: ClassVar[bool] = False

    # Every concrete class states its static type, so the abstract class has no defaults.
    @property
    @abstractmethod
    def value_shape(self) -> tuple[int | AbstractExtent, ...]: ...

    @property
    @abstractmethod
    def free_indices(self) -> tuple[tuple[Index, int | AbstractExtent], ...]: ...

    @property
    @abstractmethod
    def argument_spaces(
        self,
    ) -> tuple[AbstractFunctionSpace | tuple[AbstractFunctionSpace, ...], ...]: ...

    @property
    @abstractmethod
    def domain(self) -> AbstractDomain | AbstractTopology | None: ...

    @property
    @abstractmethod
    def scalar_type(self) -> ExpressionScalarType: ...

    @staticmethod
    def joined_scalar_type(
        *operands: AbstractExpression | tuple[AbstractExpression, ...],
    ) -> ExpressionScalarType:
        """Join Boolean values or promote numerical operands to a common scalar type."""
        types = {member.scalar_type for slot in operands for member in argument_members(slot)}
        if ScalarType.boolean in types:
            if len(types) != 1:
                raise ValueError("Boolean and numerical values cannot be mixed")
            return ScalarType.boolean
        return ScalarType.complex if ScalarType.complex in types else ScalarType.real

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

    def __getitem__(
        self, entries: int | Index | GatheredIndex | tuple[int | Index | GatheredIndex, ...]
    ) -> AbstractExpression:
        from ..tensors import component

        return component(self, argument_members(entries))

    def __iter__(self) -> Never:
        raise TypeError(f"{type(self).__name__} is not iterable")

    def __bool__(self) -> Never:
        raise TypeError("symbolic expressions have no Python truth value")
