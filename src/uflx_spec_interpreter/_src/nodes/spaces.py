from __future__ import annotations

from abc import abstractmethod
from functools import cached_property
from typing import TYPE_CHECKING, final

from .base import Base

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
        return self.space.argument_spaces


@final
class DualSpace(AbstractFunctionSpace):
    identifier = "uflx.DualSpace"
    # The antilinear forms on a space, or on the product of a tuple of spaces.
    space: AbstractFunctionSpace | tuple[AbstractFunctionSpace, ...]
    domain = None
    value_shape = ()

    @cached_property
    def scalar_type(self):
        members = self.space if isinstance(self.space, tuple) else (self.space,)
        return "complex" if any(s.scalar_type == "complex" for s in members) else "real"

    @cached_property
    def argument_spaces(self):
        # The free functions import the nodes, so the import is local.
        from ..spaces import conjugated

        return (conjugated(self.space),)
