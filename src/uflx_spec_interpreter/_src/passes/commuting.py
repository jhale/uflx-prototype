from ..substitution import freshen, map_fields


def commute(f, evaluate, avoid):
    # Move an evaluated node into its commuting operands, after making a binding node fresh.
    fresh = freshen(f, avoid)
    return map_fields(fresh, evaluate, fresh.commuting)
