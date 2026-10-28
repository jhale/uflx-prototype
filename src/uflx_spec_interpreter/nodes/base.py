from __future__ import annotations

import inspect
from abc import ABC, abstractmethod, update_abstractmethods
from typing import ClassVar, dataclass_transform, final

from ..interning import digest_of, table


@dataclass_transform(frozen_default=True, eq_default=False)
class Base(ABC):
    # The names of the construction arguments, read from the annotations.
    argument_names: ClassVar[tuple] = ()
    # Arguments that are data, not operands, so traversal skips them.
    opaque: ClassVar[tuple] = ()
    # Arguments whose nodes identify the node, which a substitution rebuilds.
    labels: ClassVar[tuple] = ()
    # A binding node names the arguments where its bound names are in scope.
    scope: ClassVar[tuple] = ()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # A concrete class of the core is final.
        for parent in cls.__bases__:
            if getattr(parent, "__final__", False):
                raise TypeError(f"{parent.__name__} is final")
        # The construction arguments are the annotated names with no value, in order.
        if "__annotate__" in cls.__dict__ and "__annotations__" not in cls.__dict__:
            raise TypeError(
                f"{cls.__name__}: a node module needs from __future__ import annotations"
            )
        names = tuple(n for n in inspect.get_annotations(cls) if n not in cls.__dict__)
        if names:
            cls.argument_names = cls.__match_args__ = names
            for k, n in enumerate(names):
                setattr(cls, n, property(lambda self, k=k: self.construction_arguments[k]))
            # A generated property may implement an abstract property.
            update_abstractmethods(cls)

    def __new__(cls, *args):
        if "identifier" not in cls.__dict__:
            raise TypeError(f"{cls.__name__} states no identifier")
        if len(args) != len(cls.argument_names):
            raise TypeError(f"{cls.__name__} takes the arguments {cls.argument_names}")
        digest = digest_of(cls.identifier, args)
        node = table.get(digest)
        if node is None:
            node = super().__new__(cls)
            vars(node).update(construction_arguments=args, digest=digest)
            table[digest] = node
        return node

    def __setattr__(self, name, value):
        raise AttributeError(f"{type(self).__name__} is immutable")

    def __hash__(self):
        return self.digest

    def __repr__(self):
        return f"{type(self).__name__}({', '.join(map(repr, self.construction_arguments))})"

    def bound(self):
        return []


class AbstractExpression(Base):
    # Every concrete class states its static type, so the abstract class has no defaults.
    @property
    @abstractmethod
    def value_shape(self) -> tuple: ...

    @property
    @abstractmethod
    def free_indices(self) -> tuple: ...

    @property
    @abstractmethod
    def arity(self) -> int: ...

    @property
    @abstractmethod
    def support(self): ...


@final
class Index(Base):
    identifier = "uflx.Index"
    label: str | tuple
    opaque = ("label",)
    labels = ("label",)
