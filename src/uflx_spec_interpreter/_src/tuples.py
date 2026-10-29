def each(fn, x):
    # fn of each member of a tuple, as a tuple, or of the one object.
    return tuple(map(fn, x)) if isinstance(x, tuple) else fn(x)


def members(x):
    # The members of a tuple, or a tuple of the one object.
    return x if isinstance(x, tuple) else (x,)
