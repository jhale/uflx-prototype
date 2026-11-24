```text
    o      _   _   _____   _
   / \    | | | | |  ___| | |     __  __
  o---o   | | | | | |_    | |     \ \/ /
 / \ / \  | |_| | |  _|   | |___   >  <
o---o---o  \___/  |_|     |_____| /_/\_\

          -- An experimental version --
```

UFLx core is a minimal language for finite element variational forms and a
potential successor to UFL, the language used by FEniCSx and Firedrake.
[UFLx core specification](https://fenics.github.io/uflx/language/) ·
[UFL](https://github.com/fenics/ufl) · [FEniCS](https://fenicsproject.org) ·
[Firedrake](https://firedrakeproject.org).

This repository contains an experimental alpha version of the language and a
basic JAX tracer/compiler.

## Installation

From the repository root, with Python 3.11 or later, run

```bash
python -m pip install -e .
python examples/poisson.py
```

## Hello world - Poisson on a triangle

The example shows how generate the stiffness and load kernels for P2 Poisson on
a triangle. 

```python
import basix
import jax
import numpy as np

from uflx.calculus import gradient, integral
from uflx.forms import form, function
from uflx.geometry import create_mesh, euclidean_space, push_forward
from uflx.passes import lower
from uflx_basix.cells import topology
from uflx_basix.quadrature import apply_quadrature
from uflx_basix.spaces import local_space
from uflx_jax_compiler import trace

# The two components of x map the reference triangle into the physical plane.
# A linear (P1) coordinate element means the three vertices determine the map.
triangle = basix.CellType.triangle

T = topology(triangle)
p1 = basix.create_element(basix.ElementFamily.P, triangle, 1)

X = local_space(T, p1, shape=(T.dimension,))
x = function(X, "x")
mesh = create_mesh(x, euclidean_space(T.dimension))

# Define quadratic (P2) trial and test functions on the reference triangle,
# then push them forward so we can take gradients on the physical triangle.
p2 = basix.create_element(basix.ElementFamily.P, triangle, 2)
V = local_space(T, p2)
u_ref, v_ref = function(V, "u"), function(V, "v")
u = push_forward(u_ref, mesh, V.law)
v = push_forward(v_ref, mesh, V.law)

a = integral(gradient(u, mesh) @ gradient(v, mesh), mesh)
L = integral(v, mesh)  # unit source

# Pull the integrals back to the reference triangle. The gradients are linear
# on this affine cell, so degree-two quadrature integrates both forms exactly.
points, weights = basix.make_quadrature(triangle, degree=2)
a_quad = apply_quadrature(lower(a), points, weights)
L_quad = apply_quadrature(lower(L), points, weights)

# Test functions index the rows of A and trial functions index its columns.
# Leaving x as a kernel input lets us supply the vertex coordinates at runtime.
a_cell = form(a_quad, (v_ref, u_ref))
L_cell = form(L_quad, (v_ref,))
stiffness = jax.jit(trace(a_cell, x))
load = jax.jit(trace(L_cell, x))

vertices = np.array([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)])
A = stiffness(vertices)
b = load(vertices)
print("Cell stiffness matrix", A)
print("Cell load vector", b)
```

The script prints the stiffness matrix and the load vector for a unit source.
See the [Full Poisson example](examples/poisson.py)

## Further examples

- Pair a Raviart–Thomas flux with a discontinuous scalar field on a curved triangle. [Mixed Poisson](examples/mixed_poisson.py)
- Generate the nine kernels needed for a DG Poisson discretisation on a mixed topology and mixed geometry mesh. [Discontinuous Galerkin Poisson](examples/dg_poisson.py)
- Differentiate a reaction–diffusion optimal control problem to build tangent, adjoint and Hessian forms. [PDE-constrained optimisation](examples/pde_constrained.py)
- Interpolate the rotation along triangle edges to compute the reduced Reissner-Mindlin shear energy. [Durán–Liberman shear reduction](examples/duran_liberman.py)

## Implementation

- `uflx` implements the full core language specification and lowers expressions to the representation a downstream form compiler consumes. [Core normal form](https://fenics.github.io/uflx/language/evaluation/#the-core-normal-form)
- `uflx_basix` supplies finite elements, basis tabulations and oriented reference facet maps. [Basix](https://github.com/fenics/basix)
- `uflx_jax_compiler` traces both original and lowered expressions to JAX callables. This gives compiler authors a reference result and lets the tests check symbolic derivatives against JAX automatic differentiation. The compiler is feature complete, but benchmarks typically run about an order of magnitude slower than FFCx. PRs related to compiler performance will not be merged.
- `uflx_extensions` exercises the core extension mechanism with complex differentiation and external operators.

### Hashing modulo alpha-equivalence

Consider two symbolic indices with `i = index("i")` and `j = index("j")`,
using `index` from `uflx.tensors`. For a vector `a`, form the sums
`s_i = index_sum(a[i], (i,))` and `s_j = index_sum(a[j], (j,))`.

In the implementation, `s_i is s_j` is `False` because the two expressions
contain distinct index nodes, `i is not j`, i.e. they are not structurally
equal. 

However, `s_i` and `s_j` are alpha-equivalent, because they differ only by the
names of their bound variables - changing the names of the variables does not
affect the mathematical meaning of the expressions.

The implementation provides an algorithm `uflx.algorithms.alpha_equivalent`
which returns `True` when two expressions with bound arguments are
alpha-equivalent.

The algorithm also applies to `component_tensor`, `form` and
`directional_derivative`, the other binding constructions.

The algorithm comes from Krzysztof Maziarz et al. [*Hashing Modulo
Alpha-Equivalence* (PLDI 2021)](https://doi.org/10.1145/3453483.3454088).

This repository uses an LLM-generated Python port of the summary code in
`AlphaHashOptimizedHash.hs` from the authors’ [Haskell
implementation](https://github.com/microsoft/hash-modulo-alpha) · [Python
port](src/uflx/_src/algorithms/_alpha_summary.py)

## Statistics

As of 23rd November 2026.

### Lines of code

| Package | Source lines |
|---|---:|
| uflx | 4,853 |
| uflx_basix | 330 |
| uflx_extensions | 100 |
| uflx_jax_compiler | 668 |
| Total | 5,951 |

### Complexity

Radon is a static analysis tool that measures cyclomatic complexity of Python
code. `uflx_jax_compiler` makes heavy use of `match` when tracing, which
perhaps unfairly scores its cyclomatic complexity.

Functions with cyclomatic complexity above 15 should be carefully considered.

| Radon score | uflx | uflx_basix | uflx_extensions | uflx_jax_compiler | Total | Distribution |
|---|---:|---:|---:|---:|---:|:---|
| 1 | 308 | 15 | 9 | 28 | 360 | ██████████████████ |
| 2 | 87 | 5 | 2 | 19 | 113 | █████▋ |
| 3 | 43 | 0 | 0 | 6 | 49 | ██▌ |
| 4 | 30 | 1 | 0 | 5 | 36 | █▊ |
| 5 | 13 | 1 | 0 | 8 | 22 | █▏ |
| 6–10 | 32 | 4 | 1 | 5 | 42 | ██▏ |
| 11–20 | 12 | 2 | 0 | 1 | 15 | ▊ |
| 21–40 | 2 | 0 | 0 | 0 | 2 | ▏ |
| 41+ | 0 | 0 | 0 | 1 | 1 | ▏ |
| Total | 527 | 28 | 12 | 73 | 640 |  |

## Caveats

This is an alpha release and is not designed for production use. The UFLx core
language is intentionally minimal - higher level layers will be added once the
core language is further developed.

## License

This project is licensed under the [MIT License](LICENSE).
