from uflx._src.passes import gradients
from uflx._src.passes.contracts import in_core_normal_form
from uflx.calculus import coordinate, gradient


def test_projection_remains():
    # On a surface the gradient of the coordinates is the projection, which needs a cell.
    from .surface import M

    pi = gradient(coordinate(M), M)
    assert gradients.apply_gradients(pi) == pi
    assert in_core_normal_form(pi)
