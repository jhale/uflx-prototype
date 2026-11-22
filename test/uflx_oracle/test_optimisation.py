"""Oracle evaluation of optimised forms."""

import pytest

pytest.importorskip("egglog")

import numpy as np

from uflx._src.optimisation import optimise
from uflx.arithmetic import num


def test_complex_grouped_form_and_rebind():
    from uflx._src.algorithms.binding import alpha_equivalent
    from uflx.arithmetic import conjugate
    from uflx.forms import form, function, rebind
    from uflx_basix.spaces import space
    from uflx_oracle.interpreter import Interpreter

    V = space("optimisation-complex", scalar_type="complex")
    u, v, w, p = (function(V, name) for name in ("u", "v", "w", "p"))
    original = form(u * conjugate(v) + u * conjugate(w), ((u, conjugate(v)),))
    result = optimise(original).expression
    assert result.arguments == original.arguments
    np.testing.assert_allclose(
        Interpreter({w: 3j}).evaluate(original, arguments=((2j, 1 + 4j),)),
        Interpreter({w: 3j}).evaluate(result, arguments=((2j, 1 + 4j),)),
    )
    a = form(num(2) * u + num(3) * u, (u,))
    assert alpha_equivalent(optimise(a).expression, optimise(rebind(a, {u: p})).expression)
