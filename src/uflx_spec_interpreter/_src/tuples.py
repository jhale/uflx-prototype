def members(x):
    # The members of a tuple, or a tuple of the one object.
    return x if isinstance(x, tuple) else (x,)
