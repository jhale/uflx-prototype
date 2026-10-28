import numpy as np

from uflx_spec_interpreter._src.substitution import expressions

from .evaluation import evaluate
from .inputs import reads_input


def fold(e):
    # The values of the largest subgraphs that read no kernel input, computed once with NumPy,
    # such as the geometry of a mesh whose coordinates are constants.
    folded, seen = {}, set()

    def visit(n):
        if n in seen:
            return
        seen.add(n)
        if not reads_input(n):
            folded[n] = np.asarray(evaluate(n, {}, np))
            return
        for c in expressions(n):
            visit(c)

    visit(e)
    return folded
