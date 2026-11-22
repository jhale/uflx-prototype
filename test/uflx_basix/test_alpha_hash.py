"""Alpha fingerprints preserve lexical identity and support cache reuse."""

from uflx._src.algorithms.alpha import alpha_hash
from uflx._src.algorithms.binding import alpha_equivalent
from uflx._src.algorithms.substitution import rename_bound
from uflx.forms import form, rebind

from ..uflx.test_alpha_hash import assert_same
from .test_alpha_equivalence import geometry


def test_outer_dependent_metadata():
    x, y, c = geometry()
    original = form(c, (x,))
    renamed = rename_bound(original, {x: y})
    assert_same(original, renamed)
    assert not alpha_equivalent(original.expression, renamed.expression)
    assert alpha_hash(original.expression) != alpha_hash(renamed.expression)


def test_rebind_geometry_fingerprints():
    x, y, c = geometry()
    original = form(c, (x,))
    renamed = rebind(original, {x: y})
    assert_same(original, renamed)
    assert alpha_hash(original.expression) != alpha_hash(renamed.expression)
