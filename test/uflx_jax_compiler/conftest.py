"""Run the numerical reference checks in double precision."""

import jax
import pytest


@pytest.fixture(autouse=True)
def reference_precision(request):
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", getattr(request, "param", True))
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)
