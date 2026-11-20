from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, final

from .arguments import argument_members
from .base import Base
from .caching import cached
from .scalar_types import NumericalScalarType, ScalarType

if TYPE_CHECKING:
    from .domains import AbstractDomain
    from .indices import AbstractExtent
    from .topology import AbstractTopology


class AbstractFunctionSpace(Base):
    @property
    @abstractmethod
    def domain(self) -> AbstractDomain | AbstractTopology | None: ...

    @property
    @abstractmethod
    def value_shape(self) -> tuple[int | AbstractExtent, ...]: ...

    @property
    @abstractmethod
    def scalar_type(self) -> NumericalScalarType: ...

    @property
    @abstractmethod
    def argument_spaces(
        self,
    ) -> tuple[AbstractFunctionSpace | tuple[AbstractFunctionSpace, ...], ...]: ...


@final
class ConjugateSpace(AbstractFunctionSpace):
    identifier = "uflx.ConjugateSpace"
    space: AbstractFunctionSpace

    @property
    def domain(self) -> AbstractDomain | AbstractTopology | None:
        return self.space.domain

    @property
    def value_shape(self) -> tuple[int | AbstractExtent, ...]:
        return self.space.value_shape

    @property
    def scalar_type(self) -> NumericalScalarType:
        return self.space.scalar_type

    @property
    def argument_spaces(
        self,
    ) -> tuple[AbstractFunctionSpace | tuple[AbstractFunctionSpace, ...], ...]:
        from ..spaces import conjugated

        return tuple(conjugated(V) for V in self.space.argument_spaces)

    def _validate(self) -> None:
        from ..spaces import trivial_conjugation
        from .validation import require

        require(
            not trivial_conjugation(self.space) and (not isinstance(self.space, ConjugateSpace)),
            "conjugate space is not canonical",
        )


@final
class DualSpace(AbstractFunctionSpace):
    identifier = "uflx.DualSpace"
    # The antilinear forms on a space, or on the product of a tuple of spaces.
    space: AbstractFunctionSpace | tuple[AbstractFunctionSpace, ...]

    @property
    def domain(self) -> None:
        return None

    @property
    def value_shape(self) -> tuple[()]:
        return ()

    @property
    @cached
    def scalar_type(self) -> NumericalScalarType:
        is_complex = any(s.scalar_type == ScalarType.complex for s in argument_members(self.space))
        return ScalarType.complex if is_complex else ScalarType.real

    @property
    @cached
    def argument_spaces(
        self,
    ) -> tuple[AbstractFunctionSpace | tuple[AbstractFunctionSpace, ...], ...]:
        # The free functions import the nodes, so the import is local.
        from ..spaces import conjugated

        return (conjugated(self.space),)

    def _validate(self) -> None:
        from .validation import require

        require(not isinstance(self.space, DualSpace), "dual space is not canonical")
        if isinstance(self.space, tuple):
            require(
                len(self.space) >= 2 and len({V.scalar_type for V in self.space}) == 1,
                "dual product needs at least two spaces with one scalar type",
            )
