from ..calculus import VALUE, piola
from ..domains import is_cell
from ..nodes import (
    Absolute,
    Action,
    ComponentTensor,
    Conjugate,
    Coordinate,
    Curl,
    DecomposedDomain,
    Divergence,
    Division,
    DomainMap,
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
    PushForward,
    RealPart,
    SquareRoot,
    Sum,
    Tensor,
    Zero,
)
from ..static import downstream, free_indices, shape, support, terminal
from ..substitution import bottom_up, everywhere, memoized
from ..tensors import identity
from .derivative_table import derivative_of_node
from .pullbacks import expand_pull_backs
from .pullbacks import remains as pulled_back_remains


def rule(partial, f, P, j):
    # Entry j of the last axis of Gradient(f, P), with the leaf rows of apply_gradients first.
    match f:
        case _ if support(f) is None:
            return Zero(
                shape(f),
                (*free_indices(f), (j, P.ambient_coordinate_domain.dimension)),
                f.argument_spaces,
            )
        case Coordinate() if P.dimension == P.ambient_coordinate_domain.dimension:
            i = Index(("uflx.apply_gradients", f, P))
            return ComponentTensor(
                Indexed(identity(P.ambient_coordinate_domain.dimension), (i, j)), (i,)
            )
        case PullBack(g, phi) if chained(f):
            # The chain rule, because a library gives the gradient of a function on a cell only.
            axes = tuple(Index(("uflx.apply_gradients", f, P, n)) for n in range(len(shape(f))))
            r = Index(("uflx.apply_gradients", f, P, "r"))
            outer = Indexed(PullBack(Gradient(g, phi.ambient_domain), phi), (*axes, r))
            inner = Indexed(partial(phi.parameterisation, P, j), (r,))
            entry = IndexSum(Product(outer, inner), (r,))
            return ComponentTensor(entry, axes) if axes else entry
        case _ if remains(f, P):
            axes = tuple(Index(("uflx.apply_gradients", f, P, n)) for n in range(len(shape(f))))
            entry = Indexed(Gradient(f, P), (*axes, j))
            return ComponentTensor(entry, axes) if axes else entry
    return derivative_of_node(f, lambda b: partial(b, P, j), lambda c: support(c) is None, [])


def gradient(f, P, partial):
    j = Index(("uflx.apply_gradients", f, P, "j"))
    axes = tuple(Index(("uflx.apply_gradients", f, P, n)) for n in range(len(shape(f))))
    df = partial(f, P, j)
    return ComponentTensor(Indexed(df, axes) if axes else df, (*axes, j))


def chained(f):
    # A pull-back by a domain map that is not a cell, whose gradient is the chain rule.
    return isinstance(f, PullBack) and isinstance(f.domain, DomainMap) and not is_cell(f.domain)


def remains(f, P):
    # The gradient of a terminal, a pull-back, a gradient, a push-forward, or a divergence or a
    # curl of Piola remains, except the gradient of the coordinates, which is the identity, and
    # the chain rule.
    identity = isinstance(f, Coordinate) and P.dimension == P.ambient_coordinate_domain.dimension
    leaf = terminal(f) or isinstance(f, PullBack | Gradient | PushForward | Divergence | Curl)
    return leaf and not identity and not chained(f)


def apply_gradients(f):
    # Inner gradients first, then the rule for the expression of each gradient.
    partial = memoized(rule)

    def evaluate(e):
        match e:
            case Gradient(g, P) if not remains(g, P):
                return gradient(g, P, partial)
            case Divergence(g, P) | Curl(g, P) if not piola(e):
                return VALUE[type(e)](g, P)
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
    PushForward,
)


def allowed(e):
    # The table of the core normal form, the output grammar N6.
    match e:
        case _ if isinstance(e, ALWAYS) or downstream(e):
            return True
        case Gradient(f, P):
            return remains(f, P)
        case Divergence() | Curl():
            return piola(e)
        case Integral(_, P):
            return not isinstance(P, DecomposedDomain)
        case PullBack(f, m):
            return pulled_back_remains(f, m)
        case Action(Function(space, _), _, _):
            return isinstance(space, DualSpace)
    return False


def lower(f):
    # apply_gradients and expand_pull_backs until neither rewrites, the stage of the core after
    # expand_integrals. A graph that the passes leave unchanged is the same object. Gradients on
    # the domains of the forms come first, so a gradient of the coordinates is the identity.
    while (g := expand_pull_backs(apply_gradients(f))) is not f:
        f = g
    return f


def in_n6(e):
    return everywhere(e, allowed)
