from uflx_spec_interpreter._src.nodes import DualSpace, Function
from uflx_spec_interpreter._src.substitution import operands


def contains(e, predicate):
    # Memoised, since the expressions share subexpressions.
    seen = {}

    def walk(n):
        if n not in seen:
            seen[n] = predicate(n) or any(walk(c) for c in operands(n))
        return seen[n]

    return walk(e)


def assert_clean(counts):
    for key in ("value", "static", "grammar"):
        assert counts.get(key, 0) == 0, counts
    if "lazy" in counts:
        assert counts["rejected"] == counts["lazy"] > 0, counts


def pairing(action):
    return isinstance(action.form, Function) and isinstance(action.form.space, DualSpace)


def static_type(e):
    # The free indices compare as a dict, since a pass may reorder them.
    return e.value_shape, dict(e.free_indices), e.argument_spaces
