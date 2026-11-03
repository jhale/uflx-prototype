from .nodes import Complex, Float, Integer, Zero


# The free functions read the static type of an expression.
def shape(e):
    return e.value_shape


def free(e):
    return dict(e.free_indices)


def arity(e):
    return len(e.argument_spaces)


def support(e):
    return e.domain


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


def field_terminal(e):
    # Literal terminals have their own evaluation rows; fields may remain unevaluated.
    return e.is_terminal and not isinstance(e, (Integer, Float, Complex, Zero))


def complete_indices(original, replacement):
    """Restore a rewrite's lost family indices, rejecting incompatible signatures.

    Promote real replacements into the original complex arithmetic before
    restoring indices. Call before rebuilding a parent that binds an index.
    """
    if (
        shape(original) != shape(replacement)
        or original.argument_spaces != replacement.argument_spaces
    ):
        raise ValueError("index completion requires matching shapes and argument spaces")
    if replacement.domain not in (None, original.domain):
        raise ValueError("a replacement changes domain")
    expected, actual = free(original), free(replacement)
    if any(i not in expected or expected[i] != n for i, n in actual.items()):
        raise ValueError("a replacement introduces an index or changes its extent")
    replacement = promote(original, replacement)
    if actual.keys() == expected.keys():
        return replacement
    from .arithmetic import add

    return add(replacement, zero_like(original, original.argument_spaces))


def promote(original, replacement):
    from .arithmetic import multiply
    from .nodes import Complex, Integer, ScalarType

    if original.scalar_type == ScalarType.complex and replacement.scalar_type == ScalarType.real:
        return multiply(Complex(Integer(1), Integer(0)), replacement)
    if original.scalar_type != replacement.scalar_type:
        raise ValueError("a rewrite changes scalar type")
    return replacement
