"""Concrete checks of mesh support versus explicit integration geometry."""

from uflx._src.algorithms.substitution import substitute
from uflx._src.meshes import cell_map
from uflx._src.nodes import Mesh
from uflx.arithmetic import num
from uflx.forms import function
from uflx.geometry import jacobian, mesh
from uflx.nodes import EuclideanSpace, ScalarType
from uflx.tensors import index
from uflx_basix.cells import reference_simplex
from uflx_basix.spaces import space

from ..testutil import table
from .meshes import grouped_topology, vertex_mesh


def test_mesh_identity_and_geometry_follow_coordinate_replacement():
    K = reference_simplex(1)
    T = grouped_topology((K,), (((0, 1),),), ())
    V = space("coordinate", value_shape=(1,), domain=T, scalar_type=ScalarType.real)
    x, y = function(V, "x"), function(V, "y")
    D, E = mesh(x, EuclideanSpace(1)), mesh(y, EuclideanSpace(1))
    assert isinstance(D, Mesh)
    assert D.construction_arguments == (x, EuclideanSpace(1))
    assert mesh(x, EuclideanSpace(1)) is D
    assert substitute(D, {x: y}) is E
    assert substitute(jacobian(cell_map(D, 0)), {x: y}) is jacobian(cell_map(E, 0))


def test_mesh_rejects_free_coordinate_indices_but_keeps_symbolic_coordinates():
    import pytest

    K = reference_simplex(1)
    T = grouped_topology((K,), (((0, 1),),), ())
    D, _ = vertex_mesh(T, [(0,), (1,)])
    q = index("configuration")
    coordinates = table([2, 3])[q] * D.coordinate_function
    with pytest.raises(ValueError, match="no free indices"):
        mesh(coordinates, EuclideanSpace(1))
    with pytest.raises(ValueError, match="no free indices"):
        Mesh(coordinates, EuclideanSpace(1))
    # A symbolic scalar parameter is fine; only an unbound family index is excluded.
    scale = function(space("scale", scalar_type=ScalarType.real), "scale")
    symbolic = mesh(scale * D.coordinate_function, EuclideanSpace(1))
    assert substitute(symbolic, {scale: num(2)}) is mesh(
        num(2) * D.coordinate_function, EuclideanSpace(1)
    )
    with pytest.raises(ValueError, match="no free indices"):
        substitute(symbolic, {scale: table([2, 3])[q]})
