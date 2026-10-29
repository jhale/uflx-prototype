from ..domains import is_cell
from ..nodes import (
    Absolute,
    Action,
    ComponentTensor,
    Conjugate,
    Coordinate,
    DecomposedDomain,
    Division,
    DualSpace,
    Float,
    Form,
    Function,
    Gradient,
    ImaginaryPart,
    ImaginaryUnit,
    Index,
    Indexed,
    IndexSum,
    Integer,
    Integral,
    Inverse,
    Logarithm,
    Normal,
    Power,
    Product,
    PullBack,
    RealPart,
    SideMap,
    SquareRoot,
    Sum,
    Tensor,
    Zero,
)
from ..static import free_indices, shape, support
from ..substitution import bottom_up, everywhere, memoized
from ..tensors import identity
from .derivative_table import derivative_of_node
from .pullbacks import expand_pull_backs, kept

# Expressions whose gradient a library provides, so the gradient remains.
LEAVES = (Function, Normal, PullBack, Gradient, Coordinate)


def rule(partial, f, P, j):
    # Entry j of the last axis of Gradient(f, P), with the leaf rows of apply_gradients first.
    match f:
        case _ if support(f) is None:
            return Zero(
                shape(f),
                (*free_indices(f), (j, P.ambient_coordinate_domain.dimension)),
                f.argument_spaces,
            )
        case Coordinate(Q) if Q == P and P.dimension == P.ambient_coordinate_domain.dimension:
            i = Index(("uflx.apply_gradients", f, P))
            return ComponentTensor(
                Indexed(identity(P.ambient_coordinate_domain.dimension), (i, j)), (i,)
            )
        case _ if isinstance(f, LEAVES):
            axes = tuple(Index(("uflx.apply_gradients", f, P, n)) for n in range(len(shape(f))))
            entry = Indexed(Gradient(f, P), (*axes, j))
            return ComponentTensor(entry, axes) if axes else entry
    return derivative_of_node(f, lambda b: partial(b, P, j), lambda c: support(c) is None, [])


def gradient(f, P, partial):
    j = Index(("uflx.apply_gradients", f, P, "j"))
    axes = tuple(Index(("uflx.apply_gradients", f, P, n)) for n in range(len(shape(f))))
    df = partial(f, P, j)
    return ComponentTensor(Indexed(df, axes) if axes else df, (*axes, j))


def evaluated(f, P):
    # A gradient remains on a leaf, except the gradient of the coordinates, which is the identity.
    if isinstance(f, Coordinate):
        return f == Coordinate(P) and P.dimension == P.ambient_coordinate_domain.dimension
    return not isinstance(f, LEAVES)


def apply_gradients(f):
    # Inner gradients first, then the rule for the expression of each gradient.
    partial = memoized(rule)

    def evaluate(e):
        match e:
            case Gradient(g, P) if evaluated(g, P):
                return gradient(g, P, partial)
        return e

    return bottom_up(f, evaluate)


# The node classes that the core normal form allows always.
ALWAYS = (
    Integer,
    Float,
    ImaginaryUnit,
    Zero,
    Sum,
    Product,
    Division,
    Power,
    Absolute,
    Conjugate,
    RealPart,
    ImaginaryPart,
    SquareRoot,
    Logarithm,
    Tensor,
    Indexed,
    ComponentTensor,
    IndexSum,
    Inverse,
    Coordinate,
    Normal,
    Function,
    Form,
)


def downstream(e):
    # A terminal of a downstream library, whose class is not a class of the core.
    return not type(e).identifier.startswith("uflx.")


def terminal(e):
    return isinstance(e, (Coordinate, Function, Normal)) or downstream(e)


def allowed(e):
    # The table of the core normal form, the output grammar N6.
    match e:
        case _ if isinstance(e, ALWAYS) or downstream(e):
            return True
        case Gradient(f, P):
            return (terminal(f) or isinstance(f, (PullBack, Gradient))) and not evaluated(f, P)
        case Integral(_, P):
            return not isinstance(P, DecomposedDomain)
        case PullBack(f, m):
            return (
                isinstance(m, SideMap)
                or terminal(f)
                or (isinstance(f, PullBack) and kept(f.domain))
                or (isinstance(f, Gradient) and not is_cell(m))
            )
        case Action(Function(space, _), _, _):
            return isinstance(space, DualSpace)
    return False


def lower(f):
    # The passes of the core after expand_integrals. Gradients on the domains of the forms come
    # first, so a gradient of the coordinates is the identity, not K times J.
    return expand_pull_backs(apply_gradients(expand_pull_backs(apply_gradients(f))))


def in_n6(e):
    return everywhere(e, allowed)
