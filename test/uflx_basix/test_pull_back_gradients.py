import basix

from uflx._src.algorithms.traversal import everywhere
from uflx._src.nodes import Inverse
from uflx.arithmetic import num
from uflx.calculus import gradient
from uflx.geometry import pull_back
from uflx.nodes import EuclideanSpace
from uflx_basix.cells import cell

from .meshes import square

K = cell(basix.CellType.triangle)
D, _, _, GEOMETRY = square(K)


def nodes(e):
    found = []
    everywhere(e, lambda n: found.append(n) or True)
    return found


def test_constant_gradient_preserves_embedding_shape():
    from uflx._src.nodes import Zero
    from uflx.geometry import cell_map, create_mesh
    from uflx.passes import expand_pull_backs
    from uflx.tensors import tensor

    embedded = create_mesh(
        tensor((D.coordinate_function[0], D.coordinate_function[1], num(0))), EuclideanSpace(3)
    )
    original = pull_back(gradient(num(2j), embedded), cell_map(embedded, 0))
    result = expand_pull_backs(original)
    assert isinstance(result, Zero)
    assert result.value_shape == (3,)
    assert result.free_indices == original.free_indices
    assert result.scalar_type == "complex"


def test_reference_constant_with_piola_mapping_is_not_a_physical_constant():
    from uflx._src.nodes import Zero
    from uflx.geometry import cell_map, push_forward, transformation_law
    from uflx.passes import expand_pull_backs
    from uflx.tensors import tensor

    mapped = push_forward(tensor((num(1), num(2))), D, transformation_law(("covariant",), 0))
    result = expand_pull_backs(pull_back(gradient(mapped, D), cell_map(D, 0)))
    assert not isinstance(result, Zero)
    assert any(isinstance(n, Inverse) for n in nodes(result))
