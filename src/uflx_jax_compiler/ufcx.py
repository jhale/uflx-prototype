"""Emit a real UFCx cell integral from a quadrature-lowered UFLx expression."""

from math import prod
from string import Template

import numpy as np
from jax import export

from ._finite_elements import basix_space
from .emitc import emit_c


def with_ufcx_cell_inputs(kernel, *, dtype):
    """Adapt a cell kernel to UFCx coordinate_dofs, coefficients (w), and constants (c)."""
    dtype = np.dtype(dtype)
    if dtype not in (np.float32, np.float64):
        raise ValueError("UFCx cell dtype must be float32 or float64")

    coordinates, *operands = kernel.inputs
    space = basix_space(coordinates.space)
    if space is None or len(space.cell_dofs) != 1:
        raise ValueError("expected a single-cell coordinate field")

    (gdim,) = coordinates.value_shape
    if not 1 <= gdim <= 3:
        raise ValueError("UFCx coordinate_dofs have three components per coordinate DOF")
    num_coordinate_dofs = space.num_dofs // gdim

    if any(op.scalar_type != "real" for op in kernel.inputs):
        raise ValueError("expected real operands for the UFCx cell integral")

    sizes = {"coordinate_dofs": 3 * num_coordinate_dofs, "w": 0, "c": 0}
    layout = []
    for op in operands:
        space = basix_space(getattr(op, "space", None))
        if space is None:
            if op.domain is not None:
                raise ValueError("expected a domain-independent UFCx constant")

            buffer, shape = "c", op.value_shape
        else:
            if len(space.cell_dofs) != 1:
                raise ValueError("expected a single-cell coefficient field")

            buffer, shape = "w", (space.num_dofs,)

        start = sizes[buffer]
        sizes[buffer] += prod(shape)
        layout.append((buffer, slice(start, sizes[buffer]), shape))

    def evaluate_from_buffers(coordinate_dofs, w, c):
        buffers = {"coordinate_dofs": coordinate_dofs, "w": w, "c": c}
        for name, array in buffers.items():
            size = sizes[name]
            if array.shape != (size,):
                raise ValueError(f"expected {name} shape {(size,)}, got {array.shape}")
            if array.dtype != dtype:
                raise TypeError(f"expected {name} dtype {dtype}, got {array.dtype}")

        coordinate_dofs = coordinate_dofs.reshape(num_coordinate_dofs, 3)[:, :gdim]
        values = (buffers[name][span].reshape(shape) for name, span, shape in layout)
        result = kernel(coordinate_dofs, *values)
        if not np.issubdtype(result.dtype, np.floating):
            raise TypeError("expected a real floating-point cell tensor")
        return result.astype(dtype)

    return evaluate_from_buffers


def emit_ufcx_tabulate_tensor(exported: export.Exported, name):
    """Return C source implementing UFCx tabulate_tensor for a cell integral.

    The kernel takes three statically sized flat arrays (coordinate_dofs, w, c)
    and returns one cell tensor, all with the same float32 or float64 dtype.
    The generated tabulate_tensor function accumulates the cell tensor into A.
    """
    if len(exported.in_avals) != 3 or len(exported.out_avals) != 1:
        raise ValueError("expected three packed inputs and one cell tensor")

    result = exported.out_avals[0]
    c_type = {np.dtype("float32"): "float", np.dtype("float64"): "double"}.get(result.dtype)
    if c_type is None or any(a.ndim != 1 or a.dtype != result.dtype for a in exported.in_avals):
        raise ValueError("expected flat inputs and a common float32 or float64 dtype")

    kernel = f"{name}_cell"
    return emit_c(exported, kernel) + _tabulate_tensor_definition(exported, name, kernel, c_type)


def _tabulate_tensor_definition(exported, name, kernel, c_type):
    # Bufferization treats these inputs as read-only; EmitC omits the C qualifier.
    names = ("coordinate_dofs", "w", "c")
    arguments = [f"({c_type} *){names[i]}" for i in exported.module_kept_var_idx]
    arguments.append("result.tensor")

    result = exported.out_avals[0]
    dimensions = result.shape or (1,)  # EmitC passes a scalar result by pointer.

    return _TABULATE_TENSOR_TEMPLATE.substitute(
        name=name,
        scalar=c_type,
        dimensions="".join(f"[{n}]" for n in dimensions),
        count=result.size,
        kernel=kernel,
        arguments=", ".join(arguments),
    )


_TABULATE_TENSOR_TEMPLATE = Template("""
void ${name}(
    ${scalar} *restrict A,
    const ${scalar} *restrict w,
    const ${scalar} *restrict c,
    const ${scalar} *restrict coordinate_dofs,
    const int *restrict entity_local_index,
    const uint8_t *restrict quadrature_permutation,
    void *custom_data) {
  // The cell kernel writes a tensor; UFCx accumulates its row-major entries.
  union {
    ${scalar} tensor${dimensions};
    ${scalar} flat[${count}];
  } result;

  ${kernel}(${arguments});
  for (size_t i = 0; i < ${count}; ++i) {
    A[i] += result.flat[i];
  }
}
""")
