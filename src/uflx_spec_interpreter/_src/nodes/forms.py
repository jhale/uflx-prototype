from __future__ import annotations

from typing import TYPE_CHECKING, Literal, final

from .arguments import _argument_members, _map_argument
from .arithmetic import Conjugate
from .caching import _cached
from .expressions import AbstractExpression
from .spaces import AbstractFunctionSpace, DualSpace

if TYPE_CHECKING:
    from .domains import AbstractDomain
    from .indices import AbstractExtent, Index
    from .topology import AbstractTopology
    from .types import _Construction


# Shared implementation helpers; the package public API is declared in __init__.py.
__all__ = ["Action", "Form", "Function", "_entry_function"]


def _entry_function(m: Function | Conjugate) -> Function:
    # The function v of an entry v or conjugate(v).
    f = m.operand if isinstance(m, Conjugate) else m
    if not isinstance(f, Function):
        raise ValueError("form arguments must be functions or their conjugates")
    return f


def _entry_space(
    entry: Function | Conjugate | tuple[Function | Conjugate, ...],
) -> AbstractFunctionSpace | tuple[AbstractFunctionSpace, ...]:
    # The space of v, the conjugate of the space for conjugate(v), and a tuple for a tuple entry.
    from ..spaces import _conjugated

    def space(m: Function | Conjugate) -> AbstractFunctionSpace:
        return _conjugated(_entry_function(m).space) if isinstance(m, Conjugate) else m.space

    return _map_argument(space, entry)


@final
class Function(AbstractExpression):
    value_fields = ()
    is_terminal = True
    identifier = "uflx.Function"
    space: AbstractFunctionSpace
    label: str | tuple[_Construction, ...]
    labels = ("label",)

    @property
    @_cached
    def value_shape(self) -> tuple[int | AbstractExtent, ...]:
        return self.space.value_shape

    @property
    @_cached
    def argument_spaces(
        self,
    ) -> tuple[AbstractFunctionSpace | tuple[AbstractFunctionSpace, ...], ...]:
        return self.space.argument_spaces

    @property
    @_cached
    def scalar_type(self) -> Literal["real", "complex"]:
        return self.space.scalar_type

    @property
    @_cached
    def domain(self) -> AbstractDomain | AbstractTopology | None:
        return self.space.domain

    @property
    @_cached
    def free_indices(self) -> tuple[tuple[Index, int | AbstractExtent], ...]:
        return ()


@final
class Form(AbstractExpression):
    value_fields = ("expression",)
    identifier = "uflx.Form"
    expression: AbstractExpression
    arguments: tuple[Function | Conjugate | tuple[Function | Conjugate, ...], ...]
    scope = ("expression", "arguments")

    def bound(self) -> tuple[Function, ...]:
        # The functions of the entries, without the conjugate of an entry conjugate(v).
        return tuple(_entry_function(m) for a in self.arguments for m in _argument_members(a))

    @property
    @_cached
    def value_shape(self) -> tuple[int | AbstractExtent, ...]:
        return self.expression.value_shape

    @property
    @_cached
    def argument_spaces(
        self,
    ) -> tuple[AbstractFunctionSpace | tuple[AbstractFunctionSpace, ...], ...]:
        # The arguments of the expression come first, then the entries.
        return self.expression.argument_spaces + tuple(map(_entry_space, self.arguments))

    @property
    @_cached
    def free_indices(self) -> tuple[tuple[Index, int | AbstractExtent], ...]:
        return self.expression.free_indices

    @property
    @_cached
    def scalar_type(self) -> Literal["real", "complex"]:
        return self.expression.scalar_type

    @property
    @_cached
    def domain(self) -> AbstractDomain | AbstractTopology | None:
        return self.expression.domain

    def _validate(self) -> None:
        from .validation import _require

        for a in self.arguments:
            if type(a) is tuple:
                _require(len(a) >= 2, "product binder needs at least two functions")
            for f in _argument_members(a):
                f = _entry_function(f)
                _require(
                    not f.argument_spaces or isinstance(f.space, DualSpace),
                    "function is not bindable",
                )
        binders = self.bound()
        _require(len(set(binders)) == len(binders), "form binders must be distinct")


@final
class Action(AbstractExpression):
    value_fields = ("form", "argument")
    identifier = "uflx.Action"
    form: AbstractExpression
    position: int
    argument: AbstractExpression | tuple[AbstractExpression, ...]

    @property
    @_cached
    def value_shape(self) -> tuple[int | AbstractExtent, ...]:
        return self.form.value_shape

    @property
    @_cached
    def argument_spaces(
        self,
    ) -> tuple[AbstractFunctionSpace | tuple[AbstractFunctionSpace, ...], ...]:
        spaces = self.form.argument_spaces
        return spaces[: self.position] + spaces[self.position + 1 :]

    @property
    @_cached
    def free_indices(self) -> tuple[tuple[Index, int | AbstractExtent], ...]:
        arguments = _argument_members(self.argument)
        from .static import _free_indices

        return _free_indices(self.form, *arguments)

    @property
    @_cached
    def scalar_type(self) -> Literal["real", "complex"]:
        return self.form.scalar_type

    @property
    @_cached
    def domain(self) -> AbstractDomain | AbstractTopology | None:
        return self.form.domain

    def _validate(self) -> None:
        from .validation import _admissible, _require

        if type(self.position) is not int:
            raise TypeError("action position must be an integer")
        _require(
            0 <= self.position < len(self.form.argument_spaces),
            "action position is outside the form",
        )
        _admissible(self.form.argument_spaces[self.position], self.argument)
