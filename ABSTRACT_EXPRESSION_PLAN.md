# AbstractExpression

> Since written, the nodes are no longer dataclasses. A class declares its
> construction arguments by annotation, `Base` interns them before
> construction, and `AbstractExpression` has no defaults, so each concrete
> class states its whole static type (see `nodes/base.py`).

A plan for splitting the toy's one base class into `Base` and
`AbstractExpression`, as the class hierarchy of LANGUAGE.md does, with the
static type of an expression as its own properties. Validation in the
constructors (rule 8) is out of scope, because it needs a check in every
class.

## Why

- `pullbacks.py` keeps `NOT_EXPRESSIONS`, a hand-written tuple of `Index`,
  `Gathered`, `Space`, the domains and the maps, so that the generic pull-back
  leaves them alone. The spec says the last row moves the pull-back into the
  operands that are expressions, which is `isinstance(x, AbstractExpression)`.
- A grammar restricts only the expression nodes (see Grammars in LANGUAGE.md),
  so the grammar checks should walk the expressions, not every object.
- `static.py` computes `shape`, `free`, `arity` and `support` by a `match` over
  every class, memoised from outside by `memoised`, and `free` returns a fresh
  `dict` because a caller may change it. In the spec the static type is
  derived from the construction arguments, and a free function only reads it.
- `domains.py` imports `free` inside two functions, because `static.py`
  imports `domains.py`. Reading the static type from the expression removes
  the import.
- The generic rule of `free` skips `Fn` by hand, because the label of a
  renamed function holds nodes. A terminal has no free indices by its own
  rule, so the skip goes.

## The hierarchy

`Node` becomes `Base`, which keeps the interning, the `identifier`, `opaque`,
`labels`, `scope` and `bound`. `AbstractExpression` extends `Base`.

| Extends `AbstractExpression` | Extends `Base` |
| ---------------------------- | -------------- |
| `Num`, `Zero`, `Sum`, `Prod`, `Div`, `Power`, `Abs`, `Conj`, `Re`, `Im`, `Sqrt`, `Log`, `Tensor`, `Indexed`, `CT`, `ISum`, `Inv` | `Index`, `Gathered` |
| `Coordinate`, `Gradient`, `D`, `Integral`, `PullBack`, `Normal` | `CoordDomain`, `Decomposed`, `DomainMap`, `CellMap`, `SideMap` |
| `Fn`, `Form`, `Action`, `Gateaux` | `Space` |

## The static type

As built, after review: each class states its own static type, so that a class
added by a language extension carries its static type with it.

1. `AbstractExpression` has four `functools.cached_property` defaults, which
   hold for most classes, and computes each once for each interned node:
   - `value_shape` is `()`;
   - `free_indices` is the union of those of the expression operands, as a
     `tuple` of the pairs of an `Index` and its extent;
   - `arity` is that of the first expression operand, or 0;
   - `support` is the support found among the expression operands.
1. A class overrides only what differs, as a short property next to its fields,
   such as `Gradient.value_shape` or `Integral.free_indices`. `Sqrt`, `Log`,
   `Abs` and the other scalar classes need no override.
1. The domains state `ambient_dim` and `topdim`, the maps state
   `parameter_domain` and `free_indices`, and `Gathered` states its free
   indices, so the expressions read them with no import of `domains.py`.
1. `shape`, `free`, `arity` and `support` in `static.py` read the properties.
   `free(e)` returns `dict(e.free_indices)`, a fresh `dict`, so no caller
   changes.

Moving the rules found that the arity of a `Form` ignored the arity of its
expression, and that `RealPart` and the other scalar classes forced arity 0.

## Using the split

| File | Change |
| ---- | ------ |
| `nodes.py` | `Base` and `AbstractExpression`, and the parent of each class from the table |
| `domains.py` | The parent of each class; `free_indices` read from `piece.psi`, with no import of `static.py` |
| `static.py` | The four readers, `free_indices` and `zero_like`; no rules and no `memoised` |
| `substitution.py` | `expressions(e)`, the operands of `e` that are expressions |
| `passes/pullbacks.py` | `pointwise` maps `pb` over the expressions; `NOT_EXPRESSIONS` goes |
| `passes/*.py` | `in_n1` to `in_n6` walk `expressions(e)` |
| `test/test_hierarchy.py` | New tests, below |

## Tests

- Every class of the left column of the table is an `AbstractExpression`, and
  no class of the right column is.
- The static type is computed once: `e.value_shape is e.value_shape`, and
  `free(e)` returns a new `dict` each time.
- The pull-back of an `Indexed` with a `Gathered` entry leaves the entry
  unchanged, which `NOT_EXPRESSIONS` guaranteed before.
- The whole suite and the negative controls pass, with the time compared to
  1 minute 30 seconds before the change.
