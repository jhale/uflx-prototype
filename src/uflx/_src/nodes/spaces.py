# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, final

from ..typing import Argument, NumericalScalarType
from .base import Base
from .caching import cached

if TYPE_CHECKING:
    from ..typing.nodes import ArgumentSpaces
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
    def argument_spaces(self) -> ArgumentSpaces: ...


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
    def argument_spaces(self) -> ArgumentSpaces:
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
    space: Argument[AbstractFunctionSpace]

    @property
    def domain(self) -> None:
        return None

    @property
    def value_shape(self) -> tuple[()]:
        return ()

    @property
    @cached
    def scalar_type(self) -> NumericalScalarType:
        from .validation import product_scalar_type

        return product_scalar_type(self.space)

    @property
    @cached
    def argument_spaces(self) -> ArgumentSpaces:
        # The free functions import the nodes, so the import is local.
        from ..spaces import conjugated

        return (conjugated(self.space),)

    def _validate(self) -> None:
        from .validation import require

        require(not isinstance(self.space, DualSpace), "dual space is not canonical")
        if isinstance(self.space, tuple):
            from .validation import product_scalar_type

            product_scalar_type(self.space)
