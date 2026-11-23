"""General reference-cell interpolation and its construction constraints."""

import basix
import pytest

from uflx.arithmetic import num
from uflx.calculus import coordinate
from uflx.nodes import EuclideanSpace
from uflx.tensors import tensor
from uflx_basix.cells import cell
from uflx_basix.interpolation import interpolate
from uflx_basix.spaces import local_space

from ..uflx.library import static_topology


def test_incompatible_inputs_are_rejected():
    K = cell(basix.CellType.triangle)
    V = local_space(K, basix.create_element(basix.ElementFamily.P, K.cell_type, 1))
    topology = static_topology((K,), ((0, 1, 2),))
    with pytest.raises(ValueError, match="reference cell"):
        interpolate(num(1), local_space(topology, V.element(0)))
    with pytest.raises(ValueError, match="value shape"):
        interpolate(coordinate(K), V)
    with pytest.raises(ValueError, match="domain"):
        interpolate(coordinate(EuclideanSpace(2))[0], V)
    with pytest.raises(NotImplementedError, match="derivative DOFs"):
        H = local_space(K, basix.create_element(basix.ElementFamily.Hermite, K.cell_type, 3))
        interpolate(num(1), H)


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
    K = cell(cell_type)
    V = local_space(K, basix.create_element(family, cell_type, degree), shape=shape)
    x = coordinate(K)

    def expression(shape, offset=1):
        if not shape:
            return num(offset) * (num(1) + x[0] ** num(2) + x[K.dimension - 1])
        return tensor(tuple(expression(shape[1:], offset * (i + 1)) for i in range(shape[0])))

    f = expression(V.value_shape)
    result = interpolate(f, V)
    return K, V, f, result


@interpolation_cases
def test_interpolation_static_type(cell_type, family, degree, shape):
    K, V, f, result = interpolation_case(cell_type, family, degree, shape)
    assert result.domain == K and result.value_shape == V.value_shape
    assert result.argument_spaces == () and result.scalar_type == f.scalar_type
