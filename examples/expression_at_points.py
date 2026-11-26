# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT

# %% [markdown]
"""# Evaluate a squared P1 field on the reference interval"""

# %%
import basix
import jax
import numpy as np

from uflx.forms import function
from uflx.tensors import index
from uflx_basix.cells import cell
from uflx_basix.evaluation import evaluate_at_points
from uflx_basix.spaces import local_space
from uflx_extensions.runtime_arrays import runtime_array, runtime_extent
from uflx_jax_compiler import trace

K = cell(basix.CellType.interval)
p1 = basix.create_element(basix.ElementFamily.P, K.cell_type, 1)
u = function(local_space(K, p1), "u")
dofs = np.array([2.0, 4.0])  # u(x) = 2 + 2x

# %% [markdown]
# ## Fixed reference points

# %%
fixed_values = evaluate_at_points(u * u, np.array([[0.25], [0.75]]), index("q"))
sample_fixed = jax.jit(trace(fixed_values, u))
print(sample_fixed(dofs))  # [6.25, 12.25]

# %% [markdown]
# ## Runtime reference points

# %%
points = runtime_array("points", (runtime_extent("n"), K.dimension))
runtime_values = evaluate_at_points(u * u, points, index("q"))
sample_runtime = jax.jit(trace(runtime_values, u, points))
print(sample_runtime(dofs, np.array([[0.0], [0.5], [1.0]])))  # [4, 9, 16]
