from .nodes import ConjugateSpace, DualSpace, ScalarType


def conjugate_space(V):
    return V if V.scalar_type == ScalarType.real else ConjugateSpace(V)


def dual_space(S):
    return DualSpace(S)


def conjugated(S):
    # The conjugate of an argument space, so that conjugating twice gives the space back.
    if isinstance(S, tuple):
        return tuple(conjugated(V) for V in S)
    return S.space if isinstance(S, ConjugateSpace) else conjugate_space(S)
