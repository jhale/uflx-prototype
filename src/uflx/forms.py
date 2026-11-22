"""Functions, function spaces, form arguments and application."""

from ._src.algorithms.form_binding import partial, rebind
from ._src.forms import apply, form, function
from ._src.spaces import conjugate_space, dual_space

__all__ = [
    "apply",
    "conjugate_space",
    "dual_space",
    "form",
    "function",
    "partial",
    "rebind",
]
