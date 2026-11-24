# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT
"""Check node metadata and private storage without repeating dataclass generation."""

import ast
from dataclasses import fields, is_dataclass
from pathlib import Path

import pytest

from uflx._src import nodes


def declaration_errors(source):
    errors = []
    metadata = {"value_fields", "scope", "binding_fields", "labels", "index_fields", "commuting"}
    for cls in ast.parse(source).body:
        if not isinstance(cls, ast.ClassDef):
            continue
        methods = {node.name: node for node in cls.body if isinstance(node, ast.FunctionDef)}
        stored = {
            node.target.id
            for node in cls.body
            if isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.value is None
        }
        declarations = {
            target.id: node.value
            for node in cls.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        if "identifier" not in declarations:
            continue
        names = {name.removeprefix("_") for name in stored}
        if {"__init__", "__new__"} & methods.keys() or "__match_args__" in declarations:
            errors.append(f"{cls.name}: constructor and positional patterns belong to dataclasses")
        for field in metadata & declarations.keys():
            members = ast.literal_eval(declarations[field])
            if len(set(members)) != len(members) or not set(members) <= names:
                errors.append(f"{cls.name}.{field}: repeated or unknown field")
        scope = set(ast.literal_eval(declarations["scope"])) if "scope" in declarations else set()
        binding = (
            set(ast.literal_eval(declarations["binding_fields"]))
            if "binding_fields" in declarations
            else set()
        )
        if "scope" in declarations and not binding <= scope:
            errors.append(f"{cls.name}: declarations must belong to their binding scope")
        if (scope or "bound" in methods) and not binding:
            errors.append(f"{cls.name}: binding requires declaration fields")
        for name in stored:
            if not name.startswith("_"):
                continue
            getter = methods.get(name[1:])
            expected = ast.parse(f"return self.{name}").body[0]
            if (
                getter is None
                or len(getter.body) != 1
                or ast.dump(getter.body[0]) != ast.dump(expected)
            ):
                errors.append(f"{cls.name}.{name}: missing or incorrect public getter")
    return errors


def test_core_node_declarations_agree():
    for path in Path(nodes.__file__).parent.glob("*.py"):
        assert not (errors := declaration_errors(path.read_text())), (path.name, errors)
    for cls in vars(nodes).values():
        if isinstance(cls, type) and issubclass(cls, nodes.Base):
            assert set(cls.binding_fields) <= set(cls.scope) <= set(cls.argument_names), cls


def test_dataclasses_supply_constructors_and_patterns():
    for cls in vars(nodes).values():
        if not isinstance(cls, type) or "identifier" not in vars(cls):
            continue
        assert is_dataclass(cls)
        stored = tuple(field.name for field in fields(cls))
        assert cls.__match_args__ == stored
        assert cls.argument_names == tuple(name.removeprefix("_") for name in stored)
        assert cls.__dataclass_params__.frozen
        assert not cls.__dataclass_params__.eq


@pytest.mark.parametrize(
    "mistake",
    [
        '__match_args__ = ("second", "first")',
        'value_fields = ("frist",)',
        'value_fields = ("first", "first")',
        'scope = ("first",)\n    binding_fields = ("second",)',
        'scope = ("first",)',
        "def __init__(self, first, second):\n        pass",
        "_third: int",
        "_third: int\n    def third(self) -> int:\n        return self.first",
    ],
)
def test_declaration_check_detects_transcription_errors(mistake):
    source = f"""class Example:
    identifier = "test.Example"
    first: int
    second: int
    {mistake}
"""
    assert declaration_errors(source)
