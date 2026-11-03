from .nodes import (
    AbstractExpression,
    Action,
    Complex,
    Conjugate,
    DirectionalDerivative,
    Division,
    DualSpace,
    Form,
    Function,
    Integer,
    Product,
    ScalarType,
    Sum,
)
from .nodes.forms import entry_function
from .substitution import occurs
from .tuples import each, members


def entry_fns(entry):
    # The functions of an entry, without the conjugate of an entry conjugate(v).
    return [m.operand if isinstance(m, Conjugate) else m for m in members(entry)]


def function(V, label):
    return Function(V, label)


def form(expression, arguments):
    return Form(expression, arguments)


def action(f, position, g):
    return Action(f, position, g)


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
        TypeError: F is not an expression or x is not a function or tuple of
            functions.
        ValueError: Another structural or static-type requirement fails.

    Existence and real-linearity of the differential are mathematical
    preconditions, not checked here. General node validity is assumed;
    admissibility uses static signatures rather than proving space membership.
    """
    xs, points, entries = members(x), members(a), members(w)
    if not isinstance(F, AbstractExpression):
        raise TypeError("the differentiated value must be an expression")
    if not all(isinstance(v, Function) for v in xs):
        raise TypeError("the variable must be a function or tuple of functions")
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
    directions = each(entry_function, w)
    ws = members(directions)
    if not all(isinstance(v, Function) for v in ws):
        raise ValueError("form arguments must be functions or their conjugates")
    if len(set(ws)) != len(ws):
        raise ValueError("direction binders must be distinct")
    if len({isinstance(v, Conjugate) for v in entries}) != 1:
        raise ValueError("tuple arguments must be all functions or all conjugates")
    if len({v.space.scalar_type for v in xs}) != 1:
        raise ValueError("mixed real and complex variables are unsupported")
    if any(v.space != d.space for v, d in zip(xs, ws, strict=True)):
        raise ValueError("directions must belong to the variable spaces")
    for variable, point in zip(xs, points, strict=True):
        if not isinstance(point, AbstractExpression):
            raise ValueError("evaluation points must be expressions")
        if (
            point.value_shape != variable.value_shape
            or point.argument_spaces != variable.argument_spaces
            or point.scalar_type not in (variable.scalar_type, ScalarType.real)
            or point.domain not in (None, variable.domain)
        ):
            raise ValueError("evaluation point is not admissible for the variable space")
    extents = {}
    for expression in (F, *points):
        for index, extent in expression.free_indices:
            if index in extents and extents[index] != extent:
                raise ValueError("shared free indices must have matching extents")
            extents[index] = extent
    if any(occurs(v, e) for v in ws for e in (F, *points)):
        raise ValueError("direction binders must not be free in the expression or point")
    delta = DirectionalDerivative(F, x, a, directions)
    if xs[0].space.scalar_type == ScalarType.real:
        return Form(delta, (w,))
    i = Complex(Integer(0), Integer(1))
    from .arithmetic import multiply

    rotated = each(lambda v: multiply(i, v), directions)
    sign = i if isinstance(entries[0], Conjugate) else Product(Integer(-1), i)
    second = Product(sign, DirectionalDerivative(F, x, a, rotated))
    return Form(Product(Division(Integer(1), Integer(2)), Sum(delta, second)), (w,))
