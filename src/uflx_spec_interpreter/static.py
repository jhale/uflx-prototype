from .forms import space
from .nodes import Zero


# The free functions read the static type of an expression.
def shape(e):
    return e.value_shape


def free(e):
    return dict(e.free_indices)


def arity(e):
    return e.arity


def support(e):
    return e.support


def free_indices(*es):
    merged = {}
    for e in es:
        merged |= free(e)
    return tuple(merged.items())


def zero_like(f, n_args, *others):
    # Placeholder argument spaces, since the toy does not track the spaces of arguments.
    return Zero(shape(f), free_indices(f, *others), tuple(space("arg") for _ in range(n_args)))
