# Experimental UFLx Core

[UFLx core](https://fenics.github.io/uflx/language/) is an expressive minimal
language specification for writing finite element variational forms. It is a
potential successor to [UFL](https://github.com/fenics/ufl) which is used by
the [FEniCS Project](https://fenicsproject.org) and
[Firedrake](https://firedrakeproject.org).

This repository contains an experimental implementation of the UFLx core
[language specification](https://fenics.github.io/uflx/language/), including
lowerings to the [core normal
form](https://fenics.github.io/uflx/language/evaluation/#the-core-normal-form)
for downstream compilers to use.

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

1. `uflx` - an implementation of the core language. The implementation is
   complete with respect to the specification.
2. `uflx_basix` - uses [Basix](https://github.com/fenics/basix) to provide
   element definitions and tabulations. 
3. `uflx_jax_compiler` - a simple form 'compiler' which traces both original
   and lowered (core normal form) UFLx expressions to JAX callables. Tests can
   check lowered forms against direct evalution, and symbolic derivatives
   against JAX automatic differentiation. It can be used as as an oracle for
   other (proper) compiler efforts. Is is feature complete with respect to the
   specification and supports `uflx_extensions`.
4. `uflx_extensions` - An extension with complex differentiation and external
   operators, to test the UFLx extension concepts.

## Examples

- [P2 Poisson problem on a single triangle](examples/poisson.py)

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
unfairly scores its cyclomatic complexity.

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
