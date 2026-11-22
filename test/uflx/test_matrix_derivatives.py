"""Fixed matrix sizes and polynomial determinant derivatives at singular points."""

import pytest

from uflx.tensors import determinant, index, inverse

from .library import runtime_array, runtime_extent


@pytest.mark.parametrize("operation", [determinant, inverse])
@pytest.mark.parametrize("shape_kind", ["square", "rows", "columns"])
def test_matrix_operations_reject_symbolic_dimensions(operation, shape_kind):
    n = runtime_extent("matrix-size")
    shape = {"square": (n, n), "rows": (n, 2), "columns": (2, n)}[shape_kind]
    with pytest.raises(ValueError, match="fixed integers"):
        operation(runtime_array("matrix", shape))


@pytest.mark.parametrize("operation", [determinant, inverse])
def test_fixed_matrices_retain_symbolic_family_indices(operation):
    q, i = runtime_extent("matrix-family"), index("i")
    matrices = runtime_array("matrices", (q, 2, 2))
    result = operation(matrices[i])
    assert result.free_indices == ((i, q),)
    assert result.value_shape == (() if operation is determinant else (2, 2))
