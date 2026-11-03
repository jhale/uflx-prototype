from __future__ import annotations

from abc import abstractmethod
from functools import cached_property
from typing import TYPE_CHECKING, final

from ..tuples import members
from .base import Base
from .scalar_types import ScalarType

if TYPE_CHECKING:
    from .domains import AbstractDomain


class AbstractFunctionSpace(Base):
    @property
    @abstractmethod
    def domain(self) -> AbstractDomain | None: ...

    @property
    @abstractmethod
    def value_shape(self) -> tuple: ...

    @property
    @abstractmethod
    def scalar_type(self) -> str: ...

    @property
    @abstractmethod
    def argument_spaces(self) -> tuple: ...


@final
class ConjugateSpace(AbstractFunctionSpace):
    identifier = "uflx.ConjugateSpace"
    space: AbstractFunctionSpace

    @property
    def domain(self):
        return self.space.domain

    @property
    def value_shape(self):
        return self.space.value_shape

    @property
    def scalar_type(self):
        return self.space.scalar_type

    @property
    def argument_spaces(self):
        from ..spaces import conjugated

        return conjugated(self.space.argument_spaces)

    def _validate(self):
        from ..validation import require, typed

        typed(self.space, AbstractFunctionSpace, "conjugate requires a space")
        require(
            self.space.scalar_type == ScalarType.complex
            and (not isinstance(self.space, ConjugateSpace)),
            "conjugate space is not canonical",
        )


@final
class DualSpace(AbstractFunctionSpace):
    identifier = "uflx.DualSpace"
    # The antilinear forms on a space, or on the product of a tuple of spaces.
    space: AbstractFunctionSpace | tuple[AbstractFunctionSpace, ...]
    domain = None
    value_shape = ()

    @cached_property
    def scalar_type(self):
        is_complex = any(s.scalar_type == ScalarType.complex for s in members(self.space))
        return ScalarType.complex if is_complex else ScalarType.real

    @cached_property
    def argument_spaces(self):
        # The free functions import the nodes, so the import is local.
        from ..spaces import conjugated

        return (conjugated(self.space),)

    def _validate(self):
        from ..validation import require, tuple_of, typed

        tuple_of(members(self.space), AbstractFunctionSpace) if type(
            self.space
        ) is tuple else typed(self.space, AbstractFunctionSpace, "dual requires a space")
        require(not isinstance(self.space, DualSpace), "dual space is not canonical")
        if type(self.space) is tuple:
            require(
                len(self.space) >= 2 and len({V.scalar_type for V in self.space}) == 1,
                "dual product needs at least two spaces with one scalar type",
            )
