from ..calculus import rotation_of_gradient, trace_of_gradient
from ..domains import is_cell
from ..names import derived_index, derived_indices
from ..nodes import (
    Absolute,
    Action,
    Complex,
    ComponentTensor,
    Conjugate,
    Coordinate,
    Curl,
    Determinant,
    Divergence,
    Division,
    DomainMap,
    Float,
    Form,
    Function,
    Gradient,
    ImaginaryPart,
    Indexed,
    IndexSum,
    Integer,
    Integral,
    Inverse,
    Logarithm,
    Mesh,
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
from ..static import field_terminal, free_indices, shape, support
from ..substitution import bottom_up, everywhere, memoized
from ..tensors import identity
from .actions import apply_actions, retained
from .derivative_table import derivative_of_node
from .piola import piola_law
from .pullbacks import expand_pull_backs
from .pullbacks import remains as pulled_back_remains


def rule(partial, f, P, j):
    # Entry j of the last axis of Gradient(f, P), with the leaf rows of apply_gradients first.
    match f:
        case Action(A, p, g):
            return Action(partial(A, P, j), p, g)
        case _ if support(f) is None:
            return Zero(
                shape(f),
                (*free_indices(f), (j, P.ambient_coordinate_domain.dimension)),
                f.argument_spaces,
                f.scalar_type,
            )
        case Coordinate() if P.dimension == P.ambient_coordinate_domain.dimension:
            i = derived_index("uflx.apply_gradients", f, P)
            return ComponentTensor(
                Indexed(identity(P.ambient_coordinate_domain.dimension), (i, j)), (i,)
            )
        case PullBack(g, phi) if chained(f):
            # The chain rule, because a library gives the gradient of a function on a cell only.
            axes = derived_indices("uflx.apply_gradients", (f, P), len(shape(f)))
            r = derived_index("uflx.apply_gradients", f, P, "r")
            outer = Indexed(PullBack(Gradient(g, phi.ambient_domain), phi), (*axes, r))
            inner = Indexed(partial(phi.parameterisation, P, j), (r,))
            entry = IndexSum(Product(outer, inner), (r,))
            return ComponentTensor(entry, axes) if axes else entry
        case _ if remains(f, P):
            axes = derived_indices("uflx.apply_gradients", (f, P), len(shape(f)))
            entry = Indexed(Gradient(f, P), (*axes, j))
            return ComponentTensor(entry, axes) if axes else entry
    return derivative_of_node(f, lambda b: partial(b, P, j), lambda c: support(c) is None, [])


def gradient(f, P, partial):
    j = derived_index("uflx.apply_gradients", f, P, "j")
    axes = derived_indices("uflx.apply_gradients", (f, P), len(shape(f)))
    df = partial(f, P, j)
    return ComponentTensor(Indexed(df, axes) if axes else df, (*axes, j))


def chained(f):
    # A pull-back by a domain map that is not a cell, whose gradient is the chain rule.
    return isinstance(f, PullBack) and isinstance(f.map, DomainMap) and not is_cell(f.map)


def remains(f, P):
    # Field and spatial-operator gradients remain, except the coordinate identity
    # and the chain rule.
    identity = isinstance(f, Coordinate) and P.dimension == P.ambient_coordinate_domain.dimension
    leaf = field_terminal(f) or isinstance(f, PullBack | Gradient | PushForward | Divergence | Curl)
    return leaf and not identity and not chained(f)


def apply_gradients(f):
    # Inner gradients first, then the rule for the expression of each gradient.
    partial = memoized(rule)

    def evaluate(e):
        match e:
            case Divergence(g, P) if piola_law(e) is None:
                return apply_gradients(trace_of_gradient(g, P))
            case Curl(g, P) if piola_law(e) is None:
                return apply_gradients(rotation_of_gradient(g, P))
            case Gradient(g, P) if not remains(g, P):
                return apply_actions(gradient(g, P, partial))
        return e

    return bottom_up(f, evaluate)


def allowed(e):
    # The core normal form, the output grammar N6.
    match e:
        case (
            Integer()
            | Float()
            | Complex()
            | Zero()
            | Sum()
            | Product()
            | Division()
            | Power()
            | Absolute()
            | Conjugate()
            | RealPart()
            | ImaginaryPart()
            | SquareRoot()
            | Logarithm()
            | Tensor()
            | Indexed()
            | ComponentTensor()
            | IndexSum()
            | Inverse()
            | Determinant()
            | Coordinate()
            | Normal()
            | Function()
            | Form()
            | PushForward()
        ):
            return True
        case _ if field_terminal(e):
            return True
        case Divergence() | Curl():
            return piola_law(e) is not None
        case Gradient(f, P):
            return remains(f, P)
        case Integral(_, P):
            return not isinstance(P, Mesh)
        case PullBack(f, m):
            return pulled_back_remains(f, m)
        case Action(A, _, _):
            return retained(A)
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
