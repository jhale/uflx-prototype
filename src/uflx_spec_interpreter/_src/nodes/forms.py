from __future__ import annotations

from typing import TYPE_CHECKING, Literal, final

from .arguments import argument_members, map_argument
from .arithmetic import Conjugate
from .caching import cached
from .expressions import AbstractExpression
from .spaces import AbstractFunctionSpace, DualSpace

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
        return conjugated(entry_function(m).space) if isinstance(m, Conjugate) else m.space

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
    def scalar_type(self) -> Literal["real", "complex"]:
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
class Form(AbstractExpression):
    value_fields = ("expression",)
    identifier = "uflx.Form"
    expression: AbstractExpression
    arguments: tuple[Function | Conjugate | tuple[Function | Conjugate, ...], ...]
    scope = ("expression", "arguments")
    binding_fields = ("arguments",)

    def bound(self) -> tuple[Function, ...]:
        # The functions of the entries, without the conjugate of an entry conjugate(v).
        return tuple(entry_function(m) for a in self.arguments for m in argument_members(a))

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
    def scalar_type(self) -> Literal["real", "complex"]:
        return self.expression.scalar_type

    @property
    @cached
    def domain(self) -> AbstractDomain | AbstractTopology | None:
        return self.expression.domain

    def _validate(self) -> None:
        from .validation import require

        for a in self.arguments:
            if type(a) is tuple:
                require(len(a) >= 2, "product binder needs at least two functions")
            for f in argument_members(a):
                f = entry_function(f)
                require(
                    not f.argument_spaces or isinstance(f.space, DualSpace),
                    "function is not bindable",
                )
        binders = self.bound()
        require(len(set(binders)) == len(binders), "form binders must be distinct")


@final
class Action(AbstractExpression):
    value_fields = ("form", "argument")
    identifier = "uflx.Action"
    form: AbstractExpression
    position: int
    argument: AbstractExpression | tuple[AbstractExpression, ...]

    @property
    @cached
    def value_shape(self) -> tuple[int | AbstractExtent, ...]:
        return self.form.value_shape

    @property
    @cached
    def argument_spaces(
        self,
    ) -> tuple[AbstractFunctionSpace | tuple[AbstractFunctionSpace, ...], ...]:
        spaces = self.form.argument_spaces
        return spaces[: self.position] + spaces[self.position + 1 :]

    @property
    @cached
    def free_indices(self) -> tuple[tuple[Index, int | AbstractExtent], ...]:
        arguments = argument_members(self.argument)
        from .static import free_indices

        return free_indices(self.form, *arguments)

    @property
    @cached
    def scalar_type(self) -> Literal["real", "complex"]:
        return self.form.scalar_type

    @property
    @cached
    def domain(self) -> AbstractDomain | AbstractTopology | None:
        return self.form.domain

    def _validate(self) -> None:
        from .validation import admissible, require

        if type(self.position) is not int:
            raise TypeError("action position must be an integer")
        require(
            0 <= self.position < len(self.form.argument_spaces),
            "action position is outside the form",
        )
        admissible(self.form.argument_spaces[self.position], self.argument)
