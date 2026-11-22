from uflx._src.nodes import SideMap
from uflx.forms import function
from uflx.geometry import pull_back
from uflx_basix.spaces import space

from .integrals import DIAGONAL, SQUARE, D, K
from .integrals import I as interval
from .meshes import grouped_topology, side_domain, side_map


def test_a_mesh_and_its_sides_have_topologies():
    assert D.topology == SQUARE
    E = side_domain(D, DIAGONAL)
    assert E.topology == DIAGONAL and E.dimension == 1 and E.ambient_domain == D.ambient_domain


def test_a_restriction_is_a_pull_back_by_a_side_map():
    E = side_domain(D, DIAGONAL)
    u = function(space("V", domain=D), "u")
    f = pull_back(u, side_map(D, E, "+"))
    assert isinstance(f.map, SideMap) and f.domain == E and f.free_indices == ()


def test_a_side_domain_takes_the_side_on_the_topology_of_the_domain():
    # As for a mesh and a submesh, the first side is on another topology.
    other = grouped_topology((K,), (((0, 1, 2),),), ())
    S = grouped_topology(
        (interval,), (((0, 1), (1, 2)),), ((("submesh", other, 0), ("mesh", SQUARE, 0)),)
    )
    E = side_domain(D, S, "mesh")
    assert E.topology == S and E.ambient_domain == D.ambient_domain
