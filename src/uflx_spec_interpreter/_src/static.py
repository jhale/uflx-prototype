from .nodes import Coordinate, Function, Normal, Zero


# The free functions read the static type of an expression.
def shape(e):
    return e.value_shape


def free(e):
    return dict(e.free_indices)


def arity(e):
    return len(e.argument_spaces)


def support(e):
    return e.support


def free_indices(*es):
    merged = {}
    for e in es:
        merged |= free(e)
    return tuple(merged.items())


def zero_like(f, argument_spaces, *others):
    return Zero(shape(f), free_indices(f, *others), argument_spaces)


def downstream(e):
    # A terminal of a downstream library, whose class is not a class of the core.
    return not type(e).identifier.startswith("uflx.")


def terminal(e):
    return isinstance(e, (Coordinate, Function, Normal)) or downstream(e)
