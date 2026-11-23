"""Physical-cell interpolation and its construction constraints."""

import basix
import pytest

from uflx.arithmetic import num
from uflx.calculus import coordinate
from uflx.tensors import tensor
from uflx_basix.interpolation import interpolate
from uflx_basix.spaces import local_space

from .meshes import single_cell_mesh


def test_incompatible_inputs_are_rejected():
    mesh = single_cell_mesh(basix.CellType.triangle)
    K = mesh[0].reference_cell
    P1 = basix.create_element(basix.ElementFamily.P, K.cell_type, 1)
    V = local_space(mesh.topology, P1)
    with pytest.raises(ValueError, match="physical mesh cell"):
        interpolate(num(1), V)
    with pytest.raises(ValueError, match="target space"):
        interpolate(num(1), local_space(K, P1), cell=mesh[0])
    with pytest.raises(ValueError, match="value shape"):
        interpolate(coordinate(mesh), V)
    with pytest.raises(ValueError, match="domain"):
        interpolate(coordinate(K)[0], V, cell=mesh[0])
    with pytest.raises(NotImplementedError, match="derivative DOFs"):
        H = local_space(
            mesh.topology, basix.create_element(basix.ElementFamily.Hermite, K.cell_type, 3)
        )
        interpolate(num(1), H, cell=mesh[0])


interpolation_cases = pytest.mark.parametrize(
    "cell_type,family,degree,shape",
    [
        (basix.CellType.interval, basix.ElementFamily.P, 2, ()),
        (basix.CellType.pyramid, basix.ElementFamily.P, 1, ()),
        (basix.CellType.triangle, basix.ElementFamily.P, 2, (2, 2)),
        (basix.CellType.triangle, basix.ElementFamily.N1E, 1, ()),
        (basix.CellType.triangle, basix.ElementFamily.N1E, 2, ()),
        (basix.CellType.triangle, basix.ElementFamily.RT, 2, ()),
        (basix.CellType.quadrilateral, basix.ElementFamily.P, 2, ()),
        (basix.CellType.tetrahedron, basix.ElementFamily.N1E, 1, ()),
        (basix.CellType.hexahedron, basix.ElementFamily.P, 1, ()),
    ],
)


def interpolation_case(cell_type, family, degree, shape):
    mesh = single_cell_mesh(cell_type)
    V = local_space(mesh.topology, basix.create_element(family, cell_type, degree), shape=shape)
    x = coordinate(mesh)

    def expression(shape, offset=1):
        if not shape:
            return num(offset) * (num(1) + x[0] ** num(2) + x[mesh.dimension - 1])
        return tensor(tuple(expression(shape[1:], offset * (i + 1)) for i in range(shape[0])))

    f = expression(V.value_shape)
    result = interpolate(f, V)
    return mesh, V, f, result


@interpolation_cases
def test_interpolation_static_type(cell_type, family, degree, shape):
    mesh, V, f, result = interpolation_case(cell_type, family, degree, shape)
    assert result.domain == mesh and result.value_shape == V.value_shape
    assert result.argument_spaces == () and result.scalar_type == f.scalar_type
