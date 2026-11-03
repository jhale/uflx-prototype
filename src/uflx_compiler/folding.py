import numpy as np

from uflx_spec_interpreter._src.traversal import _expressions, _memoized

from .evaluation import evaluate
from .inputs import arrays


def fold(e, *, preserve_indices=()):
    # The values of the largest subgraphs that read no kernel input, computed once with NumPy,
    # such as the geometry of a mesh whose coordinates are constants.
    # Preserve selected iteration axes instead of materialising their Cartesian product.
    preserved = set(preserve_indices)
    folded, reads = {}, _memoized(arrays)

    def visit(visit, n):
        if not reads(n) and not preserved.intersection(i for i, _ in n.free_indices):
            folded[n] = np.asarray(evaluate(n, {}, np))
        else:
            for c in _expressions(n):
                visit(c)

    _memoized(visit)(e)
    return folded
