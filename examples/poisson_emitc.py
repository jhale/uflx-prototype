# %% [markdown]
"""# Emitting C for Poisson

Emit a C99 stiffness kernel for P1 Poisson on an affine triangle. The script
writes the kernel and intermediate MLIR files to `output/poisson_emitc/`.
"""

# %%

import logging
from pathlib import Path

import basix
import jax
import numpy as np

import uflx_jax_compiler
from uflx.calculus import gradient, integral
from uflx.forms import form, function
from uflx.geometry import create_mesh, euclidean_space, push_forward
from uflx.passes import lower
from uflx_basix.cells import topology
from uflx_basix.quadrature import apply_quadrature
from uflx_basix.spaces import local_space

logging.basicConfig(format="%(message)s")
logging.getLogger("uflx_jax_compiler").setLevel(logging.INFO)
jax.config.update("jax_enable_x64", True)

triangle = basix.CellType.triangle
T = topology(triangle)
p1 = basix.create_element(basix.ElementFamily.P, triangle, 1)
V = local_space(T, p1)
x = function(local_space(T, p1, shape=(T.dimension,)), "x")
mesh = create_mesh(x, euclidean_space(T.dimension))

u, v = function(V, "u"), function(V, "v")
gu, gv = (gradient(push_forward(f, mesh, V.law), mesh) for f in (u, v))
a_ref = lower(integral(gv @ gu, mesh))
rule = basix.make_quadrature(triangle, 0)
a = form(apply_quadrature(a_ref, *rule), (v, u))

stiffness = jax.jit(uflx_jax_compiler.trace(a, x))
vertices = jax.ShapeDtypeStruct((T.dimension + 1, T.dimension), np.float64)
exported = jax.export.export(stiffness)(vertices)
uflx_jax_compiler.emit_c(exported, name="stiffness", output_dir=Path("output/poisson_emitc"))
