import numpy as np

from uflx_spec_interpreter._src.substitution import _expressions, _memoized

from .evaluation import evaluate
from .inputs import arrays


def fold(e):
    # The values of the largest subgraphs that read no kernel input, computed once with NumPy,
    # such as the geometry of a mesh whose coordinates are constants.
    folded, reads = {}, _memoized(arrays)

    def visit(visit, n):
        if not reads(n):
            folded[n] = np.asarray(evaluate(n, {}, np))
        else:
            for c in _expressions(n):
                visit(c)

    _memoized(visit)(e)
    return folded
