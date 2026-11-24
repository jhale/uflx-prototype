# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT

from __future__ import annotations

from typing import TYPE_CHECKING, final

from .arguments import argument_members, map_argument
from .arithmetic import Conjugate
from .caching import cached
from .expressions import AbstractExpression, BindingExpression
from .scalar_types import NumericalScalarType, numerical_type
from .spaces import AbstractFunctionSpace

if TYPE_CHECKING:
    from .domains import AbstractDomain
    from .indices import AbstractExtent, Index
    from .topology import AbstractTopology
    from .types import Construction


def entry_function(m: Function | Conjugate) -> Function:
    # The function v of an entry v or conjugate(v).
    f = m.operand if isinstance(m, Conjugate) else m
    if not isinstance(f, Function):
        raise ValueError("form arguments must be functions or their conjugates")
    return f


def _entry_space(
    entry: Function | Conjugate | tuple[Function | Conjugate, ...],
) -> AbstractFunctionSpace | tuple[AbstractFunctionSpace, ...]:
    # The space of v, the conjugate of the space for conjugate(v), and a tuple for a tuple entry.
    from ..spaces import conjugated

    def space(m: Function | Conjugate) -> AbstractFunctionSpace:
        V = entry_function(m).space
        return conjugated(V) if isinstance(m, Conjugate) else V

    return map_argument(space, entry)


@final
class Function(AbstractExpression):
    value_fields = ()
    is_terminal = True
    identifier = "uflx.Function"
    space: AbstractFunctionSpace
    label: str | tuple[Construction, ...]
    labels = ("label",)

    @property
    @cached
    def value_shape(self) -> tuple[int | AbstractExtent, ...]:
        return self.space.value_shape

    @property
    @cached
    def argument_spaces(
        self,
    ) -> tuple[AbstractFunctionSpace | tuple[AbstractFunctionSpace, ...], ...]:
        return self.space.argument_spaces

    @property
    @cached
    def scalar_type(self) -> NumericalScalarType:
        return self.space.scalar_type

    @property
    @cached
    def domain(self) -> AbstractDomain | AbstractTopology | None:
        return self.space.domain

    @property
    @cached
    def free_indices(self) -> tuple[tuple[Index, int | AbstractExtent], ...]:
        return ()


@final
class Form(BindingExpression[Function]):
    value_fields = ("expression",)
    identifier = "uflx.Form"
    expression: AbstractExpression
    arguments: tuple[Function | Conjugate | tuple[Function | Conjugate, ...], ...]
    scope = ("expression", "arguments")
    binding_fields = ("arguments",)

    @property
    @cached
    def value_shape(self) -> tuple[int | AbstractExtent, ...]:
        return self.expression.value_shape

    @property
    @cached
    def argument_spaces(
        self,
    ) -> tuple[AbstractFunctionSpace | tuple[AbstractFunctionSpace, ...], ...]:
        # The arguments of the expression come first, then the entries.
        return self.expression.argument_spaces + tuple(map(_entry_space, self.arguments))

    @property
    @cached
    def free_indices(self) -> tuple[tuple[Index, int | AbstractExtent], ...]:
        return self.expression.free_indices

    @property
    @cached
    def scalar_type(self) -> NumericalScalarType:
        return numerical_type(self.expression.scalar_type)

    @property
    @cached
    def domain(self) -> AbstractDomain | AbstractTopology | None:
        return self.expression.domain

    def _validate(self) -> None:
        from .validation import require

        for a in self.arguments:
            if type(a) is tuple:
                require(len(a) >= 2, "product binder needs at least two functions")
                require(
                    len({entry_function(member).space.scalar_type for member in a}) == 1,
                    "product binder requires one scalar type",
                )
        from ..algorithms.substitution import occurs

        binders = self.bound()
        require(len(set(binders)) == len(binders), "form binders must be distinct")
        require(
            not any(occurs(name, self.argument_spaces) for name in binders),
            "a form argument space depends on a bound function",
        )


@final
class Apply(AbstractExpression):
    value_fields = ("form", "arguments")
    identifier = "uflx.Apply"
    form: AbstractExpression
    arguments: tuple[AbstractExpression | tuple[AbstractExpression, ...], ...]

    @property
    @cached
    def value_shape(self) -> tuple[int | AbstractExtent, ...]:
        return self.form.value_shape

    @property
    @cached
    def argument_spaces(
        self,
    ) -> tuple[AbstractFunctionSpace | tuple[AbstractFunctionSpace, ...], ...]:
        return ()

    @property
    @cached
    def free_indices(self) -> tuple[tuple[Index, int | AbstractExtent], ...]:
        arguments = (m for slot in self.arguments for m in argument_members(slot))
        from .static import free_indices

        return free_indices(self.form, *arguments)

    @property
    @cached
    def scalar_type(self) -> NumericalScalarType:
        return numerical_type(self.form.scalar_type)

    @property
    @cached
    def domain(self) -> AbstractDomain | AbstractTopology | None:
        return self.form.domain

    def _validate(self) -> None:
        from .validation import admissible, require

        require(
            bool(self.arguments) and len(self.arguments) == len(self.form.argument_spaces),
            "application must supply every argument slot",
        )
        for space, value in zip(self.form.argument_spaces, self.arguments, strict=True):
            admissible(space, value)
