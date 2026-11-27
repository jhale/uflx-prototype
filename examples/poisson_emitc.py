# %% [markdown]
"""# Emitting C for Poisson

Emit a C99 stiffness kernel for P1 Poisson on an affine triangle.
"""

# %%

import basix
import jax
import numpy as np

import uflx_jax_compiler
from uflx.calculus import gradient, integral
from uflx.forms import form, function
from uflx.geometry import create_mesh, euclidean_space
from uflx.passes import lower
from uflx_basix.cells import topology
from uflx_basix.quadrature import apply_quadrature
from uflx_basix.spaces import function_space

jax.config.update("jax_enable_x64", True)

triangle = basix.CellType.triangle
T = topology(triangle)
p1 = basix.create_element(basix.ElementFamily.P, triangle, 1)
x = function(function_space(T, p1, shape=(T.dimension,)), "x")
mesh = create_mesh(x, euclidean_space(T.dimension))

V = function_space(mesh, p1)
u, v = function(V, "u"), function(V, "v")
gu, gv = (gradient(f, mesh) for f in (u, v))
a_ref = lower(integral(gv @ gu, mesh))
rule = basix.make_quadrature(triangle, 0)
a = form(apply_quadrature(a_ref, *rule), (v, u))

stiffness = jax.jit(uflx_jax_compiler.trace(a, x))
vertices = jax.ShapeDtypeStruct((T.dimension + 1, T.dimension), np.float64)
exported = jax.export.export(stiffness)(vertices)
stiffness_c = uflx_jax_compiler.emit_c(exported, "stiffness")
# stiffness_c is the C99 source, ready to compile.
