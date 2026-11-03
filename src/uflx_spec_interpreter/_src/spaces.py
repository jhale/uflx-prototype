from .nodes import AbstractFunctionSpace, ConjugateSpace, DualSpace, ScalarType


def conjugate_space(V):
    if not isinstance(V, AbstractFunctionSpace):
        raise TypeError("conjugation requires a function space")
    return (
        V
        if V.scalar_type == ScalarType.real
        else (V.space if isinstance(V, ConjugateSpace) else ConjugateSpace(V))
    )


def dual_space(S):
    return S.space if isinstance(S, DualSpace) else DualSpace(S)


def conjugated(S):
    # The conjugate of an argument space, so that conjugating twice gives the space back.
    if isinstance(S, tuple):
        return tuple(conjugated(V) for V in S)
    return S.space if isinstance(S, ConjugateSpace) else conjugate_space(S)
