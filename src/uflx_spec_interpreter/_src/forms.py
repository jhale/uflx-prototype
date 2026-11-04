from __future__ import annotations

from typing import TYPE_CHECKING

from .algorithms.substitution import occurs
from .arithmetic import add, divide, multiply
from .nodes import (
    AbstractExpression,
    Action,
    Complex,
    Conjugate,
    DirectionalDerivative,
    DualSpace,
    Form,
    Function,
    Integer,
    ScalarType,
)
from .nodes.arguments import argument_members, map_argument
from .nodes.forms import entry_function
from .nodes.validation import admissible, merge_indices

if TYPE_CHECKING:
    from .nodes.spaces import AbstractFunctionSpace
    from .nodes.types import Construction


def function(V: AbstractFunctionSpace, label: str | tuple[Construction, ...]) -> Function:
    return Function(V, label)


def form(
    expression: AbstractExpression,
    arguments: tuple[Function | Conjugate | tuple[Function | Conjugate, ...], ...],
) -> Form:
    return Form(expression, arguments)


def action(
    f: AbstractExpression, position: int, g: AbstractExpression | tuple[AbstractExpression, ...]
) -> Action:
    return Action(f, position, g)


def _validate_derivative_variables(
    x: Function | tuple[Function, ...],
    a: AbstractExpression | tuple[AbstractExpression, ...],
    w: Function | Conjugate | tuple[Function | Conjugate, ...],
) -> None:
    # Variable identity and the matching tuple structure of point and direction.
    xs, points, entries = argument_members(x), argument_members(a), argument_members(w)
    if isinstance(x, tuple) and len(x) < 2:
        raise ValueError("a tuple variable must have at least two members")
    if any(v.argument_spaces and not isinstance(v.space, DualSpace) for v in xs):
        raise ValueError("derivative variables must be bindable")
    if isinstance(x, tuple) != isinstance(a, tuple) or isinstance(x, tuple) != isinstance(w, tuple):
        raise ValueError("variable, point and argument must have matching tuple structure")
    if len(xs) != len(points) or len(xs) != len(entries):
        raise ValueError("variable, point and argument must have matching lengths")
    if len(set(xs)) != len(xs):
        raise ValueError("derivative variables must be distinct")


def _validate_derivative_directions(
    variables: tuple[Function, ...],
    entries: tuple[Function | Conjugate, ...],
    directions: tuple[Function, ...],
) -> None:
    # Each direction is a distinct binder in the corresponding variable space.
    if len(set(directions)) != len(directions):
        raise ValueError("direction binders must be distinct")
    if len({isinstance(v, Conjugate) for v in entries}) != 1:
        raise ValueError("tuple arguments must be all functions or all conjugates")
    if len({v.space.scalar_type for v in variables}) != 1:
        raise ValueError("mixed real and complex variables are unsupported")
    if any(v.space != d.space for v, d in zip(variables, directions, strict=True)):
        raise ValueError("directions must belong to the variable spaces")


def gateaux_derivative(
    F: AbstractExpression,
    x: Function | tuple[Function, ...],
    a: AbstractExpression | tuple[AbstractExpression, ...],
    w: Function | Conjugate | tuple[Function | Conjugate, ...],
) -> Form:
    """Construct a derivative form from real directional derivatives.

    For real variables, bind the real differential. For complex variables, a
    plain entry selects its complex-linear part and a conjugate entry selects
    its complex-antilinear part. The new argument follows F's existing slots.
    This constructor does not test holomorphicity or perform differentiation.

    Args:
        F: Expression to differentiate, with geometry held fixed.
        x: Bindable function, or at least two distinct bindable functions.
            Tuple members must be all real or all complex.
        a: Evaluation point, matching x's tuple structure. Each member must
            have the variable's value shape and argument spaces, its scalar
            type or the real scalar type, and its support or no support.
            Symbolic forms are admissible points for dual variables.
        w: Fresh direction functions in the corresponding variable spaces,
            optionally conjugated. Tuple members must be distinct and either
            all plain or all conjugated. They must not occur free in F or a.

    Returns:
        An interned Form containing one directional derivative for real
        variables or the two-projection composition for complex variables.

    Raises:
        ValueError: A structural or static-signature requirement fails.

    Python argument types are checked statically.

    Existence and real-linearity of the differential are mathematical
    preconditions, not checked here. General node validity is assumed;
    admissibility uses static signatures rather than proving space membership.
    """
    _validate_derivative_variables(x, a, w)
    xs, points, entries = argument_members(x), argument_members(a), argument_members(w)
    directions = map_argument(entry_function, w)
    ws = argument_members(directions)
    _validate_derivative_directions(xs, entries, ws)
    for variable, point in zip(xs, points, strict=True):
        admissible(variable.space, point)
    merge_indices(F.free_indices, *(point.free_indices for point in points))
    if any(occurs(v, e) for v in ws for e in (F, *points)):
        raise ValueError("direction binders must not be free in the expression or point")
    delta = DirectionalDerivative(F, x, a, directions)
    if xs[0].space.scalar_type == ScalarType.real:
        return Form(delta, (w,))
    i = Complex(Integer(0), Integer(1))

    def rotate(v: Function) -> AbstractExpression:
        return multiply(i, v)

    rotated = map_argument(rotate, directions)
    sign = i if isinstance(entries[0], Conjugate) else multiply(Integer(-1), i)
    second = multiply(sign, DirectionalDerivative(F, x, a, rotated))
    half = divide(Integer(1), Integer(2))
    projected_derivative = multiply(half, add(delta, second))
    return Form(projected_derivative, (w,))
