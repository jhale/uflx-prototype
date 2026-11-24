# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT
"""Shared identity, immutable storage and binding metadata for nodes."""

from __future__ import annotations

import hashlib
import math
import weakref
from abc import ABCMeta
from dataclasses import Field, dataclass, fields
from typing import TYPE_CHECKING, ClassVar, TypeVar, cast, dataclass_transform

from .caching import cached

if TYPE_CHECKING:
    from ..typing.nodes import Construction
    from .forms import Function
    from .indices import Index


_Node = TypeVar("_Node")
_table: weakref.WeakValueDictionary[tuple[type[Base], int], Base] = weakref.WeakValueDictionary()


class _Interned(ABCMeta):
    """Intern fully initialized, validated nodes; never reinitialize a cache hit."""

    def __call__(cls: type[_Node], *args: Construction) -> _Node:

        node_class = cast("type[Base]", cls)

        if "identifier" not in node_class.__dict__:
            raise TypeError(f"{node_class.__name__} states no identifier")
        if len(args) != len(node_class.argument_names):
            raise TypeError(
                f"{node_class.__name__} takes the arguments {node_class.argument_names}"
            )
        _construction_arguments(args)
        digest = _digest_of(node_class.identifier, args)
        key = (node_class, digest)
        node = _table.get(key)
        if node is None:
            node = cast("Base", ABCMeta.__call__(cast("ABCMeta", cls), *args))
            vars(node).update(construction_arguments=args, digest=digest)
            _table[key] = node
        return cast("_Node", node)


@dataclass_transform(frozen_default=True, eq_default=False)
class Base(metaclass=_Interned):
    # Public construction order, derived from dataclass fields.
    argument_names: ClassVar[tuple[str, ...]] = ()
    # Expression operands evaluated by passes; all construction arguments remain dependencies.
    value_fields: ClassVar[tuple[str, ...]] = ()
    is_terminal: ClassVar[bool] = False
    # Index entries selected when instantiating a value, excluding identity labels.
    index_fields: ClassVar[tuple[str, ...]] = ()
    # Arguments whose nodes identify the node, which a substitution rebuilds.
    labels: ClassVar[tuple[str, ...]] = ()
    # A binding node names the arguments where its bound names are in scope.
    scope: ClassVar[tuple[str, ...]] = ()
    # Declaration operands are renamed, but are not free occurrences.
    binding_fields: ClassVar[tuple[str, ...]] = ()

    __dataclass_fields__: ClassVar[dict[str, Field[Construction]]]
    identifier: ClassVar[str]

    @property
    @cached
    def construction_arguments(self) -> tuple[Construction, ...]:
        return tuple(getattr(self, name) for name in self.argument_names)

    @property
    @cached
    def digest(self) -> int:
        return _digest_of(self.identifier, self.construction_arguments)

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        # A concrete class of the core is final.
        for parent in cls.__bases__:
            if getattr(parent, "__final__", False):
                raise TypeError(f"{parent.__name__} is final")
        # Frozen dataclasses own storage, constructors and positional patterns.
        # A private field implements a public abstract property of the same name.
        dataclass(cls, frozen=True, eq=False, repr=False)
        cls.argument_names = tuple(field.name.removeprefix("_") for field in fields(cls))

    def __post_init__(self) -> None:
        self._validate()

    def __hash__(self) -> int:
        return self.digest

    def __repr__(self) -> str:
        return f"{type(self).__name__}({', '.join(map(repr, self.construction_arguments))})"

    def _validate(self) -> None:
        pass

    def bound(self) -> tuple[Index | Function, ...]:
        return ()


def construction_equal(left: Construction, right: Construction) -> bool:
    """Compare construction data exactly, treating embedded nodes as identities.

    Unlike Python tuple equality, preserve numeric types and signed floating
    zero, as the construction digest does. Traverse tuples without recursion.
    """
    pending = [(left, right)]
    while pending:
        a, b = pending.pop()
        if a is b:
            continue
        if type(a) is not type(b):
            return False
        if isinstance(a, tuple) and isinstance(b, tuple):
            if len(a) != len(b):
                return False
            pending.extend(zip(reversed(a), reversed(b), strict=True))
        elif isinstance(a, float) and isinstance(b, float):
            if repr(a) != repr(b):
                return False
        elif a != b:
            return False
    return True


def construction_encoding(value: Construction) -> str:
    """Encode construction data with exact scalar types and opaque node identities."""
    return _encode(value)


def _encode(value: Construction) -> str:
    if isinstance(value, tuple):
        return "(" + ",".join(map(_encode, value)) + ")"
    return (
        f"#{value.digest:032x}" if isinstance(value, Base) else f"{type(value).__name__}:{value!r}"
    )


def _digest_of(identifier: str, values: tuple[Construction, ...]) -> int:
    data = _encode((identifier, values)).encode()
    return int.from_bytes(hashlib.blake2b(data, digest_size=16).digest(), "big")


def _construction_arguments(args: tuple[object, ...]) -> None:
    for value in args:
        if isinstance(value, Base):
            continue
        if type(value) is tuple:
            _construction_arguments(cast("tuple[object, ...]", value))
        elif type(value) is float:
            if not math.isfinite(value):
                raise ValueError("construction floats must be finite")
        elif type(value) not in (bool, int, str):
            raise TypeError(f"invalid construction argument type: {type(value).__name__}")
