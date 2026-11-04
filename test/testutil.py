import numpy as np

import uflx_spec_interpreter
from uflx_spec_interpreter import RD, ScalarType, image, num, tensor
from uflx_spec_interpreter._src.algorithms.traversal import _expressions, _nodes_in
from uflx_spec_interpreter._src.nodes import (
    AbstractExpression,
    Action,
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
    Mesh,
    Product,
    PullBack,
    Sum,
    Tensor,
)
from uflx_toy.nodes import VertexCoordinates


def contains(e, predicate):
    # Memoised, since the expressions share subexpressions.
    seen = {}

    def walk(n):
        if n not in seen:
            seen[n] = predicate(n) or any(
                walk(c) for c in _nodes_in(n, (*n.value_fields, *n.index_fields))
            )
        return seen[n]

    return walk(e)


def assert_clean(counts):
    for key in ("value", "static", "grammar", "typed"):
        assert counts.get(key, 0) == 0, counts
    if "lazy" in counts:
        assert counts["rejected"] == counts["lazy"] > 0, counts


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
            return ("first",)
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
        case Action():
            return ("form", "argument")
    return ()


def ill_typed(e):
    # The first node that breaks a rule of the argument spaces, which construction would raise.
    # The message names the spaces only, since the repr of a shared graph can be exponential.
    seen = {}

    def walk(n):
        if n not in seen:
            seen[n] = node_error(n) or next(filter(None, map(walk, _expressions(n))), None)
        return seen[n]

    return walk(e)


def node_error(n):
    others = [m for name in n.value_fields if name not in passed(n) for m in _nodes_in(n, [name])]
    for m in others:
        if isinstance(m, AbstractExpression) and m.argument_spaces:
            return f"{type(n).__name__} has an operand with argument spaces {m.argument_spaces}"
    match n:
        case Sum(a, b) if a.argument_spaces != b.argument_spaces:
            return f"Sum of {a.argument_spaces} and {b.argument_spaces}"
        case Tensor(entries) if len({x.argument_spaces for x in entries}) > 1:
            return f"Tensor of {[x.argument_spaces for x in entries]}"
        case Action(f, p, g):
            if p >= len(f.argument_spaces):
                return f"Action at {p} of {f.argument_spaces}"
            if spaces_of_value(f.argument_spaces[p]) != spaces_of(g):
                return f"Action at {p} of {f.argument_spaces} on {spaces_of(g)}"
            if complex_for_real(f.argument_spaces[p], g):
                return f"Action at {p} of {f.argument_spaces} on a complex argument"
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


def vertex_mesh(T, vertices, Q=None):
    # The mesh of T with the vertices at the given coordinates, mapped by degree 1 on each cell.
    vertices = tuple(tuple(float(c) for c in v) for v in np.asarray(vertices, dtype=float))
    return image(VertexCoordinates(T, vertices), RD(len(vertices[0])) if Q is None else Q)


def push_forward(f, domain, law=None):
    """Use the toy function space's law when an example omits it."""
    return uflx_spec_interpreter.push_forward(f, domain, f.space.law if law is None else law)


def side_map(domain, side, selection=None):
    """Use the first side selection when an example omits it."""
    topology = side.topology if isinstance(side, Mesh) else side
    return uflx_spec_interpreter.side_map(
        domain, side, topology.sides[0][0][0] if selection is None else selection
    )


def side_domain(domain, topology, selection=None):
    """Use the first side selection when an example omits it."""
    return uflx_spec_interpreter.side_domain(
        domain, topology, topology.sides[0][0][0] if selection is None else selection
    )
