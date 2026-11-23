# Experimental UFLx core language implementation

[UFLx core](https://fenics.github.io/uflx/language/) is an expressive minimal
language specification for writing finite element variational forms. It is a
potential successor to [UFL](https://github.com/fenics/ufl) which is used by
the [FEniCS Project](https://fenicsproject.org) and
[Firedrake](https://firedrakeproject.org).

This repository contains an experimental implementation of the [UFLx core
language specification](https://fenics.github.io/uflx/language/), including
lowerings to the [core normal
form](https://fenics.github.io/uflx/language/evaluation/#the-core-normal-form)
for typical downstream form compilers.

## Installation

```bash
pip install git+https:// 
```

## Components

This Python library contains the following packages:

1. `uflx` - an implementation of the core language. The implementation is
   complete with respect to the specification.
2. `uflx_basix` - uses [Basix](https://github.com/fenics/basix) to provide
   element definitions, tabulations, and oriented reference facet maps.
3. `uflx_jax_compiler` - a deliberately tiny form 'compiler' that traces both
   original and lowered (core normal form) UFLx expressions to JAX callables.
   Tests can check lowered forms against direct evalution, and symbolic UFLx
   derivatives against JAX automatic differentiation. Benchmarks are typically
   an order of magnitude slower than FFCx. The compiler is feature complete and
   can be used as as an oracle for other, more performant, efforts.
4. `uflx_extensions` - An extension with complex differentiation and external
   operators, intended to test the UFLx core extension concept.

## Examples

- [P2 Poisson problem on a single triangle](examples/poisson.py)
- [Mixed Poisson problem on a single triangle](examples/mixed_poisson.py)
- [Cell and interior-facet kernels for DG Poisson on curved quadratic triangles](examples/dg_poisson.py)

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

This software is alpha and is not designed for production use. The UFLx core
language is intentionally minimal - higher level layers will be added once the
core language is further developed.

## License
