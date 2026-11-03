from .nodes import Coordinate, Function, Normal, Sum, Zero


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
    return Zero(
        shape(f),
        free_indices(f, *others),
        argument_spaces,
        f.scalar_type,
    )


def downstream(e):
    # A terminal of a downstream library, whose class is not a class of the core.
    return not type(e).identifier.startswith("uflx.")


def terminal(e):
    return isinstance(e, (Coordinate, Function, Normal)) or downstream(e)


def complete_indices(original, replacement):
    """Restore a rewrite's lost family indices, rejecting incompatible signatures.

    This is only index completion, not scalar promotion or support validation.
    Call it before rebuilding a parent that binds an index.
    """
    if (
        shape(original) != shape(replacement)
        or original.argument_spaces != replacement.argument_spaces
    ):
        raise ValueError("index completion requires matching shapes and argument spaces")
    expected, actual = free(original), free(replacement)
    if any(i not in expected or expected[i] != n for i, n in actual.items()):
        raise ValueError("a replacement introduces an index or changes its extent")
    if actual.keys() == expected.keys():
        return replacement
    return Sum(replacement, zero_like(original, original.argument_spaces))
