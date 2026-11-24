```text
            o
           /|\          ____    ____   ___________   ____
          / | \        /  /|   /  /|  /         /|  /  /|
         /  |  o      #### |  #### | ########### / #### |
        /   | /|\     #### |  #### | ###########/  #### |          ____   ____
       /    |/ | \    #### |  #### | ###########   #### |         /  / \ /  /
      o-----o  |  o   #### |  #### | ####/    /|   #### |        ####   ####
     /|    /|\ | /    #### |  #### | ######### /   #### |         #### ####
    / |   / | \|/     #### /__#### / #########/    #### /______   ####### \
   /  |  /  |  o      ####/   ####/  #########     ####/      /|    #####   \
  /   | /   | /       ############   #### /        ########### /   #######   \
 /    |/    |/         ##########    ####/         ###########/   #### #### /
o-----o-----o           ########     ####          ###########   ####   ####

o------------------------- An experimental proposal -------------------------o
```

UFLx core is a minimal language writing for (finite element) variational forms.
It is a potential successor to UFL, the language used by FEniCSx and Firedrake.

[UFLx core specification](https://fenics.github.io/uflx/language/) ·
[UFL](https://github.com/fenics/ufl) · [FEniCS](https://fenicsproject.org) ·
[Firedrake](https://firedrakeproject.org).

This repository contains an alpha version of the core language and a basic JAX
tracer/compiler. The implementation is feature complete with respect to the
specification. The full repository is ~6500 lines of code plus tests.

The final design, if accepted, will be reimplemented in a structured way at
[UFLx](https://github.com/fenics/uflx).

## Installation

With Python 3.11 or later, run

```bash
git clone https://github.com/jhale/uflx-prototype.git
cd uflx-prototype
python -m pip install -e .
python examples/poisson.py
```

Dependencies are `numpy`, `jax`, `fenics-basix` and `typing-extensions`.

### Building the documentation

From the repository root, install the documentation dependencies and build the
HTML pages with Sphinx.

```bash
python -m pip install -e '.[docs]'
python -m sphinx -W -b html -c docs -d build/doctrees . build/docs
```

## Headline features

- A strict language ruleset that leads to:
    - nearly zero boiler-plate code,
    - automatic and robust hashing,
    - node identity determined entirely by the node type and its construction
      arguments,
    - no simplification on construction - the users full intent is kept in the
      first graph.
- Language features for defining arbitrarily complex topological and geometric
  configurations of cells.
- Unified constructs and rules for restriction and integration of functions on
  entities of any dimension or co-dimension.
- A single algebra system for all expressions, including forms and zeroes.
- Explicit index/tensor notation with a direct mapping to modern tensor
  contraction engines.
- Minimal use of operator overloading `+` `@` etc., except in universally
  well-understood algebraic settings.
- Explicit argument binding for form construction. In `e = conjugate(v) * u`,
  the complex scalar functions `v` and `u` are both free.
  `a = form(e, (conjugate(v), u))` binds them as arguments, declaring
  antilinearity in `v` and linearity in `u`. For a function `w` in the same
  space as `u`, `partial(a, {u: w})` leaves an antilinear form in `v`.
- Mixed forms can bind a nested tuple such as `((velocity, pressure), control)`
  as one argument. There is no separate mixed space to construct.
- A single directional derivative system, covering variational calculus on
  forms, coupled problems, scalar parameter derivatives, vector/tensor
  derivatives, symmetry, complex derivatives, and shape calculus.
- Support for scalar and tensor-valued global or restricted constants, and
  differentiating with respect to them.
- An implementation of more advanced functional analysis concepts such as
  duality, multi-linear maps (operators), and conjugates.
- Lambda-calculus inspired features for bound objects, including full (`apply`)
  and partial (`partial`) application, and rebinding (`rebind`).
- A [core normal
  form](https://fenics.github.io/uflx/language/evaluation/#the-core-normal-form)
  which specifies what a 'typical' downstream tabulator and compiler must
  accept, after the built-in `lower` pass. Downstream consumers are free to
  compose fine-grained lowerings and make other decisions.
- A clear policy on language extensions, from simple passes through to full
  language extensions. External operators and a complex derivative helper is
  currently implemented as language extensions.

## Examples

- [Poisson on a triangle](examples/poisson.py) builds the cell stiffness matrix
  and load vector for Poisson’s equation with a unit source.
- [Mixed Poisson](examples/mixed_poisson.py) couples a Raviart–Thomas flux with
  a discontinuous scalar on a curved triangle.
- [Discontinuous Galerkin Poisson](examples/dg_poisson.py) generates nine cell
  and facet kernels for a mesh with mixed cell types and geometries.
- [PDE-constrained optimisation](examples/pde_constrained.py) differentiates a
  reaction–diffusion Lagrangian to build tangent, adjoint and Hessian forms.
- [Durán–Liberman shear reduction](examples/duran_liberman.py) interpolates
  the rotation along triangle edges to compute the reduced Reissner–Mindlin
  shear energy.

### Hello world - Poisson on a triangle

The quadratic basis functions and quadrature are defined on the reference
triangle. At runtime, the three vertices passed to the kernels determine the
physical triangle.

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
from uflx_jax_compiler import basis_tables, trace

# A linear (P1) coordinate map places the reference triangle in the physical plane.
triangle = basix.CellType.triangle
T = topology(triangle)

p1 = basix.create_element(basix.ElementFamily.P, triangle, 1)
X = local_space(T, p1, shape=(T.dimension,))
x = function(X, "x")
mesh = create_mesh(x, euclidean_space(T.dimension))

# Push the quadratic (P2) trial and test functions forward to the physical triangle.
p2 = basix.create_element(basix.ElementFamily.P, triangle, 2)
V = local_space(T, p2)
u_ref, v_ref = function(V, "u"), function(V, "v")
u = push_forward(u_ref, mesh, V.law)
v = push_forward(v_ref, mesh, V.law)

# Map the stiffness and load integrals to the reference triangle.
stiffness_integral = lower(integral(gradient(u, mesh) @ gradient(v, mesh), mesh))
load_integral = lower(integral(v, mesh))

rule = basix.make_quadrature(triangle, degree=2)
points, weights = rule

# Fix the load rule on the reference integral.
L = form(apply_quadrature(load_integral, points, weights), (v_ref,))
load = jax.jit(trace(L, x))

a = form(apply_quadrature(stiffness_integral, points, weights), (v_ref, u_ref))
stiffness = jax.jit(trace(a, x))

vertices = np.array([(1.0, 1.0), (4.0, 2.0), (2.0, 5.0)])
b = load(vertices)
A = stiffness(vertices)

print("Cell load vector", b)
print("Cell stiffness matrix", A)
```

The six entries of the load vector sum to the area of this triangle, 5.5,
and the stiffness matrix is symmetric and 6 × 6. The same kernels can be
called with new vertex coordinates to evaluate another triangle.

The same stiffness form can take its quadrature rule at runtime. [Runnable Poisson example](examples/poisson.py)

```python
from uflx_extensions.runtime_arrays import RuntimeArray, RuntimeExtent

n = RuntimeExtent("n")
points_input = RuntimeArray("points", (n, 2))
weights_input = RuntimeArray("weights", (n,))
weighted_sum = apply_quadrature(stiffness_integral, points_input, weights_input)
runtime_form = form(weighted_sum, (v_ref, u_ref))
tables = basis_tables(runtime_form)
runtime_stiffness = jax.jit(trace(runtime_form, x, points_input, weights_input, *tables))

for degree in (2, 4):
    points, weights = basix.make_quadrature(triangle, degree)
    values = [table.element.tabulate(table.order, points) for table in tables]
    matrix = runtime_stiffness(vertices, points, weights, *values)
    np.testing.assert_allclose(matrix, A, rtol=1e-5, atol=1e-6)
```

## Implementation

- `uflx` implements the full core language specification and lowers expressions
  to the representation a typical downstream form compiler can consume. [Core
  normal
  form](https://fenics.github.io/uflx/language/evaluation/#the-core-normal-form)
- `uflx_basix` supplies finite elements, basis tabulations and oriented
  reference facet maps. [Basix](https://github.com/fenics/basix)
- `uflx_jax_compiler` traces both original and lowered expressions to JAX
  callables. It gives a reference result and tests UFLx's symbolic derivatives
  against JAX's automatic differentiation. The compiler is feature complete,
  but benchmarks typically run about an order of magnitude slower than FFCx.
  PRs related to compiler performance will not be merged.
- `uflx_extensions` exercises the core extension mechanism with complex
  differentiation and external operators.

## Filing issues

Open an issue in this repository with the smallest example you can give.

In-scope:

- For an interpreter issue, include a minimal working example and the
  commit you tested (`git rev-parse --short HEAD`). Give the result you
  expected and why, alongside the value or traceback you received.
- For a specification issue, write down the mathematical expression and point
  to the rule that seems ambiguous, contradictory or missing. If two readings
  give different answers, show both. [UFLx core
  specification](https://fenics.github.io/uflx/language/).
- Naming and vocabulary suggestions are encouraged.
- Issues related to API consistency and granularity.
- Discussions on language design with respect to its efficient and easy
  translation to languages used by form compilers (MLIR, JAX, C/C++ etc.).

Out-of-scope:

- No pull requests. File an issue, we then modify the language specification,
  if needed, and then we can implement the change together.
- Issues related to the compilers performance.
- Algorithms related to transformations for generating optimised form compiler
  output.
- Requests to add 'trivial' features (e.g. more elementary functions, pretty
  differential calculus constructions).


## Statistics

Measured on 24 November 2026 at commit `587d360bef` using Radon 6.0.1 on the
four Python packages under `src/`.

### Lines of code

| Package | Source lines |
|---|---:|
| uflx | 4,859 |
| uflx_basix | 473 |
| uflx_extensions | 140 |
| uflx_jax_compiler | 1,075 |
| Total | 6,547 |

### Complexity

Radon measures the cyclomatic complexity of Python code. 

The distribution counts functions and methods, including nested definitions,
and skips ellipsis-only stubs.

Functions with cyclomatic complexity above 15 should be carefully considered.

Note that `uflx_jax_compiler` makes heavy use of `match` when tracing, which
unfairly scores its cyclomatic complexity.

| Radon score | uflx | uflx_basix | uflx_extensions | uflx_jax_compiler | Total | Distribution |
|---|---:|---:|---:|---:|---:|:---|
| 1 | 309 | 15 | 12 | 31 | 367 | <img src="docs/complexity-bar.svg" width="184" height="10" alt=""> |
| 2 | 87 | 11 | 4 | 21 | 123 | <img src="docs/complexity-bar.svg" width="62" height="10" alt=""> |
| 3 | 43 | 2 | 0 | 11 | 56 | <img src="docs/complexity-bar.svg" width="28" height="10" alt=""> |
| 4 | 30 | 2 | 0 | 4 | 36 | <img src="docs/complexity-bar.svg" width="18" height="10" alt=""> |
| 5 | 13 | 2 | 0 | 9 | 24 | <img src="docs/complexity-bar.svg" width="12" height="10" alt=""> |
| 6–10 | 31 | 6 | 2 | 13 | 52 | <img src="docs/complexity-bar.svg" width="26" height="10" alt=""> |
| 11–20 | 13 | 1 | 0 | 7 | 21 | <img src="docs/complexity-bar.svg" width="11" height="10" alt=""> |
| 21–40 | 2 | 1 | 0 | 1 | 4 | <img src="docs/complexity-bar.svg" width="2" height="10" alt=""> |
| 41+ | 0 | 0 | 0 | 1 | 1 | <img src="docs/complexity-bar.svg" width="1" height="10" alt=""> |
| Total | 528 | 40 | 18 | 98 | 684 |  |

## LLM statement

I used OpenAI Astra 6.1, OpenAI Astra 6.2, and Claude Opus 5.5.

- `uflx` has been built carefully and most of the code has been reviewed,
  edited, and adjusted by hand. The majority of the complexity is in
  `uflx.algorithms`. `uflx.rules` and `uflx.passes` are pretty easy to follow,
  unlike the core language specification description.
- `uflx_basix` is LLM-generated, with some manual work done to get
  `interpolate` implemented elegantly. 
- `uflx_jax_compiler` is LLM-generated with some passes to get the API looking
  'JAX-like' and some basic performance work (vectorisation, precomputed basis
  data, reference tensors, tensor contractions, hoisting of constant terms).
- The tests. Don't look at the tests - there are lots of them, some of them are
  good, a lot are probably noise. None of them are written in a consistent or
  good style.

## License

This project is licensed under the [MIT License](LICENSE).

The hashing modulo alpha-equivalence algorithm is based on Krzysztof Maziarz et
al. [*Hashing Modulo Alpha-Equivalence* (PLDI
2021)](https://doi.org/10.1145/3453483.3454088).

This repository uses a direct LLM-generated Python port of the code in
`AlphaHashOptimizedHash.hs` from the [Haskell
implementation](https://github.com/microsoft/hash-modulo-alpha). The Copyright
remains with Microsoft under the MIT license.
