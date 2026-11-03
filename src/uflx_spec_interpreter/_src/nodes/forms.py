from __future__ import annotations

from functools import cached_property
from typing import final

from ..tuples import each, members
from .arithmetic import Conjugate
from .base import AbstractExpression
from .spaces import AbstractFunctionSpace, DualSpace


def entry_function(m):
    # The function v of an entry v or conjugate(v).
    return m.operand if isinstance(m, Conjugate) else m


def entry_space(entry):
    # The space of v, the conjugate of the space for conjugate(v), and a tuple for a tuple entry.
    from ..spaces import conjugated

    def space(m):
        return conjugated(m.operand.space) if isinstance(m, Conjugate) else m.space

    return each(space, entry)


@final
class Function(AbstractExpression):
    value_fields = ()
    is_terminal = True
    identifier = "uflx.Function"
    space: AbstractFunctionSpace
    label: str | tuple
    labels = ("label",)

    @cached_property
    def value_shape(self):
        return self.space.value_shape

    @cached_property
    def argument_spaces(self):
        return self.space.argument_spaces

    @cached_property
    def scalar_type(self):
        return self.space.scalar_type

    @cached_property
    def domain(self):
        return self.space.domain

    @cached_property
    def free_indices(self):
        return ()

    def _validate(self):
        from ..validation import typed

        typed(self.space, AbstractFunctionSpace, "function requires a space")
        typed(self.label, (str, tuple), "function label must be a string or tuple")


@final
class Form(AbstractExpression):
    value_fields = ("expression",)
    identifier = "uflx.Form"
    commuting = ("expression",)
    expression: AbstractExpression
    arguments: tuple
    scope = ("expression", "arguments")

    def bound(self):
        # The functions of the entries, without the conjugate of an entry conjugate(v).
        return [entry_function(m) for a in self.arguments for m in members(a)]

    @cached_property
    def value_shape(self):
        return self.expression.value_shape

    @cached_property
    def argument_spaces(self):
        # The arguments of the expression come first, then the entries.
        return self.expression.argument_spaces + tuple(map(entry_space, self.arguments))

    @cached_property
    def free_indices(self):
        return self.expression.free_indices

    @cached_property
    def scalar_type(self):
        return self.expression.scalar_type

    @cached_property
    def domain(self):
        return self.expression.domain

    def _validate(self):
        from ..validation import require, tuple_of, typed

        typed(self.expression, AbstractExpression, "form body must be an expression")
        tuple_of(self.arguments, (Function, Conjugate, tuple))
        for a in self.arguments:
            if type(a) is tuple:
                require(len(a) >= 2, "product binder needs at least two functions")
            for f in members(a):
                f = f.operand if isinstance(f, Conjugate) else f
                typed(f, Function, "binder must be a function or its conjugate")
                require(
                    not f.argument_spaces or isinstance(f.space, DualSpace),
                    "function is not bindable",
                )
        require(len(set(self.bound())) == len(self.bound()), "form binders must be distinct")


@final
class Action(AbstractExpression):
    value_fields = ("form", "argument")
    identifier = "uflx.Action"
    form: AbstractExpression
    position: int
    argument: AbstractExpression | tuple[AbstractExpression, ...]

    @cached_property
    def value_shape(self):
        return self.form.value_shape

    @cached_property
    def argument_spaces(self):
        spaces = self.form.argument_spaces
        return spaces[: self.position] + spaces[self.position + 1 :]

    @cached_property
    def free_indices(self):
        arguments = members(self.argument)
        return tuple(dict(p for x in (self.form, *arguments) for p in x.free_indices).items())

    @cached_property
    def scalar_type(self):
        return self.form.scalar_type

    @cached_property
    def domain(self):
        return self.form.domain

    def _validate(self):
        from ..validation import admissible, require, typed

        typed(self.form, AbstractExpression, "action requires a form expression")
        if type(self.position) is not int:
            raise TypeError("action position must be an integer")
        require(
            0 <= self.position < len(self.form.argument_spaces),
            "action position is outside the form",
        )
        admissible(self.form.argument_spaces[self.position], self.argument)
