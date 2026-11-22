# SPDX-License-Identifier: MIT
"""Translate a conservative subset of pinned Herbie rules; never execute Racket."""

from __future__ import annotations

import argparse
import hashlib
import json
import keyword
import re
import subprocess
from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVISION = "81fe5efeb5accea858051e5a53264fdc925f9a65"
type_expr = str | tuple  # Recursive S-expressions, including non-rule Racket forms.


def forms(source: str) -> Iterator[type_expr]:
    """Read balanced data, respecting strings, line comments and #; datum comments."""
    tokens = iter(re.findall(r'#;|"(?:\\.|[^"\\])*"|[()\[\]]|;[^\n]*|[^\s()\[\]";]+', source))

    def read(token):
        if token.startswith(";"):
            return None
        if token == "#;":
            for following in tokens:
                if read(following) is not None:
                    break
            else:
                raise ValueError("datum comment without a datum")
            return None
        if token in ("(", "["):
            closing = ")" if token == "(" else "]"
            values = []
            for following in tokens:
                if following == closing:
                    return tuple(values)
                value = read(following)
                if value is not None:
                    values.append(value)
            raise ValueError("unclosed datum")
        if token in (")", "]"):
            raise ValueError("mismatched delimiter")
        return token

    for token in tokens:
        value = read(token)
        if value is not None:
            yield value


def catalogue(source: str) -> Iterator[tuple[str, str, type_expr, type_expr, tuple]]:
    """Read literal top-level define-rules blocks, excluding commented-out rules."""
    for form in forms(source):
        if isinstance(form, tuple) and form and form[0] == "define-rules":
            _, group, *entries = form
            for entry in entries:
                if not isinstance(entry, tuple) or len(entry) < 3:
                    raise ValueError(f"malformed rule: {entry}")
                name, lhs, rhs, *flags = entry
                yield group, name, lhs, rhs, tuple(flags)


def number(value: type_expr) -> Fraction | None:
    """Recognise exact literals, without approximating or evaluating expressions."""
    if isinstance(value, str) and re.fullmatch(r"-?\d+(?:/\d+)?", value):
        return Fraction(value)
    return None


def variables(expression: type_expr) -> set[str]:
    if isinstance(expression, str):
        return set() if number(expression) is not None else {expression}
    return set().union(*(variables(arg) for arg in expression[1:]))


def translate(expression: type_expr) -> str:
    """Translate polynomial arithmetic; reject unproved domain restrictions."""
    if (value := number(expression)) is not None:
        return f"Scalar.number(BigRat({value.numerator}, {value.denominator}))"
    if isinstance(expression, str):
        if not re.fullmatch(r"[a-z][a-z0-9]*", expression) or keyword.iskeyword(expression):
            raise ValueError(f"unsupported symbol: {expression}")
        return expression
    op, *args = expression
    if op == "neg" and len(args) == 1:
        return f"({translate('-1')} * {translate(args[0])})"
    if op in ("+", "*", "-") and len(args) == 2:
        a, b = map(translate, args)
        return f"({a} + ({translate('-1')} * {b}))" if op == "-" else f"({a} {op} {b})"
    if op == "/" and len(args) == 2:
        divisor = number(args[1])
        if divisor is None or divisor == 0:
            raise ValueError("division requires a nonzero literal denominator")
        return f"({translate(args[0])} / {translate(args[1])})"
    if op == "pow" and len(args) == 2:
        exponent = number(args[1])
        if exponent is None or exponent.denominator != 1 or not 1 <= exponent <= 8:
            raise ValueError("power requires a literal integer exponent in 1..8")
        # Use products so these identities match the differentiated polynomial too.
        factors = [translate(args[0])] * int(exponent)
        result = factors[0]
        for factor in factors[1:]:
            result = f"({result} * {factor})"
        return result
    raise ValueError(f"unsupported operator or arity: {op}/{len(args)}")


def generate(source: str, licence: str) -> tuple[str, dict]:
    """Return standalone generated rules and a complete inclusion/exclusion report."""
    report = {
        "revision": REVISION,
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "rules": [],
    }
    groups = {"arithmetic": [], "polynomials": []}
    seen = set()
    for group, name, lhs, rhs, flags in catalogue(source):
        row = {"group": group, "name": name}
        try:
            if group not in groups:
                raise ValueError("group outside arithmetic/polynomials scope")
            if flags:
                raise ValueError(f"unsupported rule flags: {flags}")
            if isinstance(lhs, str) and number(lhs) is None:
                raise ValueError("bare-variable left-hand side")
            left, right = translate(lhs), translate(rhs)
            if variables(rhs) - variables(lhs):
                raise ValueError("right-hand side introduces a variable")
            guards = sorted(variables(lhs) - variables(rhs))
            key = left, right, tuple(guards)
            if left == right:
                raise ValueError("identity after operator translation")
            if key in seen:
                raise ValueError("duplicate after operator translation")
            seen.add(key)
            row.update(status="translated", defined_guards=guards)
            conditions = "".join(f", defined({variable})" for variable in guards)
            groups[group].append(
                (name, variables(lhs), f"yield rewrite({left}).to({right}{conditions})")
            )
        except ValueError as error:
            row.update(status="skipped", reason=str(error))
        report["rules"].append(row)
    header = "\n".join(f"# {line}".rstrip() for line in licence.strip().splitlines())
    lines = [
        header,
        "# SPDX-License-Identifier: MIT",
        f"# Generated from Herbie {REVISION}; do not edit.",
        "# Regenerate: python3 tools/translate_herbie.py",
        '"""Opt-in Herbie identities translated to UFLx scalar terms."""',
        "",
        "from collections.abc import Iterator",
        "",
        "from egglog import BigRat, RewriteOrRule, rewrite, ruleset",
        "",
        "from .language import Scalar",
        "from .rules import defined",
        "",
    ]
    for group, entries in groups.items():
        names = sorted(set().union(*(names for _, names, _ in entries)))
        parameters = ", ".join(f"{name}: Scalar" for name in names)
        lines.extend(
            [
                "",
                "@ruleset",
                f"def herbie_{group}({parameters}) -> Iterator[RewriteOrRule]:",
                f'    """Translated {group} rules with definedness guards."""',
            ]
        )
        for name, _, body in entries:
            lines.extend([f"    # Herbie: {name}", f"    {body}"])
    cleaned = subprocess.run(
        ["ruff", "check", "--fix", "--select", "UP034", "--stdin-filename", "herbie.py", "-"],
        input="\n".join(lines) + "\n",
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    formatted = subprocess.run(
        ["ruff", "format", "--stdin-filename", "herbie.py", "-"],
        input=cleaned,
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    return formatted, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    vendor = ROOT / "third_party/herbie"
    code, report = generate((vendor / "rules.rkt").read_text(), (vendor / "LICENSE.md").read_text())
    outputs = {
        ROOT / "src/uflx/_src/optimisation/herbie.py": code,
        vendor / "translation.json": json.dumps(report, indent=2) + "\n",
    }
    for path, content in outputs.items():
        if args.check:
            if not path.exists() or path.read_text() != content:
                raise SystemExit(f"out of date: {path}")
        else:
            path.write_text(content)
    included = sum(r["status"] == "translated" for r in report["rules"])
    print(f"{included}/{len(report['rules'])} upstream rules translated")


if __name__ == "__main__":
    main()
