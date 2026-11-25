# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT
"""JAX evaluation and tracing of UFLx expressions."""

from ._basis_tables import basis_tables
from ._geometry_tables import extract_geometry
from .emitc import emit_c, lowering_stages
from .interpreter import reference_trace, trace
from .ufcx import emit_ufcx_tabulate_tensor, with_ufcx_cell_inputs

__all__ = [
    "basis_tables",
    "emit_c",
    "emit_ufcx_tabulate_tensor",
    "extract_geometry",
    "lowering_stages",
    "reference_trace",
    "trace",
    "with_ufcx_cell_inputs",
]
