# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT

from uflx._src.nodes import (
    AbstractExpression,
    Apply,
    ComponentTensor,
    Conjugate,
    ConjugateSpace,
    DirectionalDerivative,
    Division,
    DualSpace,
    Form,
    Function,
    Gradient,
    Indexed,
    IndexSum,
    Integral,
    Product,
    PullBack,
    Sum,
    Tensor,
)
from uflx._src.nodes.structure import expression_operands, nodes_in
from uflx.arithmetic import num
from uflx.forms import apply, form, function
from uflx.nodes import ScalarType
from uflx.tensors import tensor


def contains(e, predicate):
    # Memoised, since the expressions share subexpressions.
    seen = {}

    def walk(n):
        if n not in seen:
            seen[n] = predicate(n) or any(
                walk(c) for c in nodes_in(n, (*n.value_fields, *n.index_fields))
            )
        return seen[n]

    return walk(e)


def assert_clean(counts):
    for key in ("value", "static", "grammar", "typed"):
        assert counts.get(key, 0) == 0, counts
    if "lazy" in counts:
        assert counts["rejected"] == counts["lazy"] > 0, counts


def form_with_fixed_argument(operator, position, value):
    """Generate a form over the remaining slots of a random operator."""
    arguments = []
    declarations = []
    members = value if isinstance(value, tuple) else (value,)
    label = ("test.fixed", operator.digest, position, tuple(v.digest for v in members))
    for slot, space in enumerate(operator.argument_spaces):
        if slot == position:
            arguments.append(value)
            continue
        if isinstance(space, tuple):
            entry = tuple(function(V, (*label, slot, i)) for i, V in enumerate(space))
        else:
            entry = function(space, (*label, slot))
        declarations.append(entry)
        arguments.append(entry)
    body = apply(operator, tuple(arguments))
    return form(body, tuple(declarations)) if declarations else body


def pairing(action):
    return isinstance(action.form, Function) and isinstance(action.form.space, DualSpace)


def keeps_static_type(e, out):
    # The free indices compare as a dict, since a pass may reorder them.
    same = (e.value_shape, dict(e.free_indices), e.argument_spaces) == (
        out.value_shape,
        dict(out.free_indices),
        out.argument_spaces,
    )
    # Rewrites preserve scalar arithmetic, but may lose the domain of a constant.
    scalar = out.scalar_type == e.scalar_type
    return same and scalar and out.domain in (e.domain, None)


def passed(n):
    # The operands whose argument spaces a node passes on, by the static type table.
    match n:
        case Sum() | Product():
            return ("first", "second")
        case Division():
            return ("numerator",)
        case Conjugate():
            return ("operand",)
        case Tensor():
            return ("entries",)
        case (
            Indexed()
            | ComponentTensor()
            | IndexSum()
            | Gradient()
            | DirectionalDerivative()
            | Integral()
            | PullBack()
            | Form()
        ):
            return ("expression",)
        case Apply():
            return ("form", "arguments")
    return ()


def ill_typed(e):
    # The first node that breaks a rule of the argument spaces, which construction would raise.
    # The message names the spaces only, since the repr of a shared graph can be exponential.
    seen = {}

    def walk(n):
        if n not in seen:
            seen[n] = node_error(n) or next(filter(None, map(walk, expression_operands(n))), None)
        return seen[n]

    return walk(e)


def node_error(n):
    others = [m for name in n.value_fields if name not in passed(n) for m in nodes_in(n, [name])]
    for m in others:
        if isinstance(m, AbstractExpression) and m.argument_spaces:
            return f"{type(n).__name__} has an operand with argument spaces {m.argument_spaces}"
    match n:
        case Sum(a, b) if a.argument_spaces != b.argument_spaces:
            return f"Sum of {a.argument_spaces} and {b.argument_spaces}"
        case Tensor(entries) if len({x.argument_spaces for x in entries}) > 1:
            return f"Tensor of {[x.argument_spaces for x in entries]}"
        case Apply(f, arguments):
            if len(arguments) != len(f.argument_spaces):
                return "application arity differs from its form"
            for space, value in zip(f.argument_spaces, arguments, strict=True):
                if spaces_of_value(space) != spaces_of(value) or complex_for_real(space, value):
                    return f"invalid argument for {space}"
    return None


def spaces_of_value(V):
    # The argument spaces of a value for an argument in V, or in the conjugate of V.
    if isinstance(V, tuple):
        return tuple(map(spaces_of_value, V))
    return (V.space if isinstance(V, ConjugateSpace) else V).argument_spaces


def complex_for_real(V, g):
    # A complex argument for a real space, which only a real argument may fill.
    if isinstance(V, tuple):
        return any(complex_for_real(W, h) for W, h in zip(V, g, strict=True))
    return g.scalar_type == ScalarType.complex and V.scalar_type == ScalarType.real


def spaces_of(g):
    return tuple(map(spaces_of, g)) if isinstance(g, tuple) else g.argument_spaces


def check_types(bad, *es):
    for e in es:
        error = ill_typed(e)
        if error:
            bad["typed"] = bad.get("typed", 0) + 1
            if bad["typed"] <= 2:
                print("ill typed:", error)
            return


def table(rows):
    # A constant tensor of literals with the values of nested lists.
    def build(x):
        return tensor(tuple(build(y) for y in x)) if isinstance(x, (list, tuple)) else num(x)

    return build(rows)
