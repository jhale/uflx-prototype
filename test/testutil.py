from uflx_spec_interpreter.substitution import operands


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
