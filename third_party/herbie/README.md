# Herbie rewrite rules

`rules.rkt` is an unmodified copy of `src/core/rules.rkt` from
[Herbie](https://github.com/herbie-fp/herbie), pinned to commit
[`81fe5efeb5accea858051e5a53264fdc925f9a65`](https://github.com/herbie-fp/herbie/tree/81fe5efeb5accea858051e5a53264fdc925f9a65).
`LICENSE.md` is the unmodified upstream MIT licence. The generated Python module
also includes the full copyright and permission notice so it accompanies wheels.

Regenerate from the repository root:

```sh
python3 tools/translate_herbie.py
python3 tools/translate_herbie.py --check
```

The generator uses Python and Ruff; it does not execute Racket or require Herbie
at runtime. It reads literal top-level `define-rules` blocks, respecting `;` line
comments and `#;` datum comments. It does not translate dynamically constructed
array-lowering or sound-operation-removal rules.

The output is `src/uflx_spec_interpreter/_src/optimisation/herbie.py`.
`translation.json` records the source SHA-256, revision, and disposition of every
active rule in the literal catalogue. Of 479 rules, 75 are translated (59 arithmetic,
16 polynomial). The remainder have explicit exclusion reasons.

Translation maps addition and multiplication directly; subtraction and negation
use multiplication by exact -1. Rational literals remain exact. Division requires
a nonzero literal denominator. Literal positive integer powers up to eight are
expanded into grouped products, allowing polynomial identities to match AD output.
Variable denominators, general powers, unsupported operators, flags, bare-variable
left sides, identities and duplicate translations are excluded. Rewrites retain
their upstream direction. Variables discarded by a rule require `defined` facts.
The restriction to polynomial operations and nonzero constant divisors makes
those guards sufficient to preserve definedness for this subset.

The optimiser accepts `OptimisationOptions(rule_groups=("herbie", "indexed"))`.
This substitutes the generated collection for the small `arithmetic` group;
constant folding and existing unary rules still apply. The default is unchanged.
`preserve_floating_point_evaluation=True` disables these mathematical rewrites.
Existing definedness analysis remains in force; this translation does not change
input or extension-node contracts.

This is an experimental rule collection, not polynomial canonicalisation or a
port of Herbie's accuracy optimiser. It can grow e-graphs rapidly. The iteration
and input-region limits are not memory or elapsed-time limits; use a subprocess
with a timeout for larger experiments. A richer catalogue does not imply better
extracted code.

When updating upstream, replace the two vendored files at a specific revision,
update the generator's revision, regenerate, review the inclusion/exclusion report,
and run `test/test_herbie_translation.py` alongside the optimisation tests.
