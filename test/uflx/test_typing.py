# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT
"""Check that strict analysis rejects invalid types previously checked at runtime."""

import importlib.util
import json
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest


@pytest.mark.skipif(importlib.util.find_spec("pyright") is None, reason="install the lint extra")
def test_strict_type_contracts(tmp_path):
    root = Path(__file__).resolve().parents[2]
    fixture = Path(__file__).with_name("typecheck") / "arguments.py"
    expected = {
        (line, text.split("# expect: ", 1)[1].strip())
        for line, text in enumerate(fixture.read_text().splitlines(), 1)
        if "# expect: " in text
    }
    # Analyze the deliberately invalid fixture separately; production checking excludes tests.
    settings = tomllib.loads((root / "pyproject.toml").read_text())["tool"]["pyright"]
    (tmp_path / fixture.name).write_text(fixture.read_text())
    settings.update(include=[fixture.name], exclude=[], extraPaths=[str(root / "src")])
    config = tmp_path / "pyrightconfig.json"
    config.write_text(json.dumps(settings))
    result = subprocess.run(
        [sys.executable, "-m", "pyright", "--project", str(config), "--outputjson"],
        capture_output=True,
        text=True,
        check=False,
    )
    report = json.loads(result.stdout)
    actual = {
        (item["range"]["start"]["line"] + 1, item.get("rule"))
        for item in report["generalDiagnostics"]
    }
    assert result.returncode == 1, result.stderr
    assert actual == expected, report["generalDiagnostics"]
    assert report["summary"]["filesAnalyzed"] == 1
