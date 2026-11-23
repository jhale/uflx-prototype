# Experimental UFLx Core

[UFLx core](https://fenics.github.io/uflx/language/) is an expressive minimal
language for writing finite element variational forms. It is a potential
successor to [UFL](https://github.com/fenics/ufl) which is used by the [FEniCS
Project](https://fenicsproject.org) and
[Firedrake](https://firedrakeproject.org).

This repository contains an experimental implementation of the UFLx core
[language specification](https://fenics.github.io/uflx/language/), including
lowerings to the [core normal
form](https://fenics.github.io/uflx/language/evaluation/#the-core-normal-form)
for downstream compilers.

## Installation

```bash
pip install git+https:// 
```

## Caveats

This software is alpha quality and not designed for production use. If UFLx is
pursued further, it is likely that we will re-write the code from scratch using
the specification and successful elements from this experiment. UFLx core
language is intentionally minimal - higher level layers will be added once the
core language is further developed.

## Components

1. `uflx` - an implementation of the core language specification. The
   implementation is complete with respect to the specification.
2. `uflx_basix` - uses [Basix](https://github.com/fenics/basix) to provide
   element definitions and tabulations. 
3. `uflx_jax_compiler` - a simple reference form compiler which traces original
   and lowered (core normal form) UFLx expressions to JAX callables. Tests can
   check lowered forms against direct evalution, and symbolic derivatives
   against JAX automatic differentiation. It can be used as as an oracle for
   other (proper) compiler efforts. Is is feature complete with respect to the
   specification and supports `uflx_extensions`.
4. `uflx_extensions` - An extension with complex differentiation and external
   operators, to test the UFLx extension concepts.

## Examples

### Poisson problem on triangles

```python
import basix
import numpy as np

from uflx.calculus import gradient, integral
from uflx.forms import form, function
from uflx.geometry import push_forward
from uflx.nodes import EuclideanSpace, Mesh
from uflx_basix.cells import topology
from uflx_basix.spaces import space
from uflx_jax_compiler import trace

triangle = basix.CellType.triangle
T = topology(triangle)

p1 = basix.create_element(basix.ElementFamily.P, triangle, 1)
X = space(T, p1, shape=(2,))
x = function(X, "x")
mesh = Mesh(x, EuclideanSpace(2))

p2 = basix.create_element(basix.ElementFamily.P, triangle, 2)
V = space(T, p2)
u_ref, v_ref = function(V, "u"), function(V, "v")
u = push_forward(u_ref, mesh, V.law)
v = push_forward(v_ref, mesh, V.law)

a = form(integral(gradient(u, mesh) @ gradient(v, mesh), mesh), (v_ref, u_ref))
L = form(integral(v, mesh), (v_ref,))

load = trace(L, x)

# Supply the coordinate values and the P2 test-function values.
vertices = np.array([(0.0, 0.0), (2.0, 0.0), (0.0, 1.0)])
area = load(vertices, np.ones(p2.dim))  # L(1) is the area, 1.
```

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
code. `uflx_jax_compiler` makes heavy use of `match` which unfairly scores its
cyclomatic complexity.

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

## License
