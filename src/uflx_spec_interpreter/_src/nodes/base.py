"""Shared identity, immutable storage and binding metadata for nodes."""

from __future__ import annotations

from dataclasses import Field, dataclass, fields
from typing import TYPE_CHECKING, ClassVar, dataclass_transform

from .interning import _cached, _digest_of, _Interned
from .types import _Construction

if TYPE_CHECKING:
    from .forms import Function
    from .indices import Index


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

    __dataclass_fields__: ClassVar[dict[str, Field[_Construction]]]
    identifier: ClassVar[str]

    @property
    @_cached
    def construction_arguments(self) -> tuple[_Construction, ...]:
        return tuple(getattr(self, name) for name in self.argument_names)

    @property
    @_cached
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
