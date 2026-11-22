"""Broadcasting preserves value axes, family indices and lazy empty values."""

import pytest

from uflx.nodes import Zero
from uflx.tensors import broadcast_to


@pytest.mark.parametrize("source,target", [((2,), ()), ((2,), (3,)), ((0,), (1,)), ((), (-1,))])
def test_invalid_shapes(source, target):
    with pytest.raises(ValueError):
        broadcast_to(Zero(source, (), (), "real"), target)
