# AbstractExpression

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

1. `AbstractExpression` has four properties, each a
   `functools.cached_property`, which writes to the instance `__dict__` and so
   works on a frozen dataclass:
   - `value_shape`, a `tuple`;
   - `free_indices`, a `tuple` of the pairs of an `Index` and its extent, in
     order of first occurrence, so the property is immutable;
   - `arity`, the number of arguments, because the toy does not track the
     argument spaces beyond their number;
   - `support`, a domain or `None`.
1. The rules stay in `static.py`, one `match` per property as now, renamed
   `rule_shape`, `rule_free`, `rule_arity` and `rule_support`, and read the
   static type of the operands through the properties. With interning, each
   node computes each property once, on first use.
1. `shape`, `free`, `arity` and `support` in `static.py` become readers:
   `free(e)` returns `dict(e.free_indices)`, a fresh `dict`, so no caller
   changes. `memoised` goes.

## Using the split

| File | Change |
| ---- | ------ |
| `nodes.py` | `Base` and `AbstractExpression`, and the parent of each class from the table |
| `domains.py` | The parent of each class; `free_indices` read from `piece.psi`, with no import of `static.py` |
| `static.py` | The rules read the properties of the operands; the four readers; no `memoised` |
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
