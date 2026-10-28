from core import operands


def contains(e, predicate):
    # Memoised on identity, since the expressions share subexpressions.
    seen = {}

    def walk(n):
        if id(n) not in seen:
            seen[id(n)] = predicate(n) or any(walk(c) for c in operands(n))
        return seen[id(n)]

    return walk(e)
