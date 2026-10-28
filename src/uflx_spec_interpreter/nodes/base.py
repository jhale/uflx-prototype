from __future__ import annotations

import hashlib
import inspect
import numbers
import weakref
from abc import ABC, abstractmethod, update_abstractmethods
from functools import cached_property
from typing import ClassVar, dataclass_transform, final


@dataclass_transform(frozen_default=True, eq_default=False)
class Base(ABC):
    # Every node by its digest, so that equal nodes are the same object.
    _table: ClassVar[weakref.WeakValueDictionary] = weakref.WeakValueDictionary()
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
        digest = Base.digest_of(cls.identifier, args)
        node = Base._table.get(digest)
        if node is None:
            node = super().__new__(cls)
            vars(node).update(construction_arguments=args, digest=digest)
            Base._table[digest] = node
        return node

    @staticmethod
    def encode(value):
        # A canonical text, so 1, 1.0 and True differ, and so do 0.0 and -0.0.
        match value:
            case Base():
                return f"#{value.digest:032x}"
            case tuple():
                return "(" + ",".join(map(Base.encode, value)) + ")"
            case bool() | str() | None:
                return repr(value)
            case numbers.Integral():
                return str(int(value))
            case numbers.Real():
                return float(value).hex()
            case numbers.Complex():
                return f"complex({float(value.real).hex()},{float(value.imag).hex()})"
        raise TypeError(f"not a construction argument: {value!r}")

    @staticmethod
    def digest_of(identifier, values):
        # Seed-free, unlike the built-in hash of str and bytes, so equal in every process.
        data = Base.encode((identifier, tuple(values))).encode()
        return int.from_bytes(hashlib.blake2b(data, digest_size=16).digest(), "big")

    def __setattr__(self, name, value):
        raise AttributeError(f"{type(self).__name__} is immutable")

    def __hash__(self):
        return self.digest

    def __repr__(self):
        return f"{type(self).__name__}({', '.join(map(repr, self.construction_arguments))})"

    def rebuild(self, **changes):
        # An unknown name adds an argument, which the class rejects.
        values = dict(zip(self.argument_names, self.construction_arguments)) | changes
        return type(self)(*values.values())

    def operands_of(self, names):
        # The nodes in the named arguments, through tuples, in order.
        out = []

        def walk(x):
            if isinstance(x, Base):
                out.append(x)
            elif isinstance(x, tuple):
                for y in x:
                    walk(y)

        for name in names:
            walk(getattr(self, name))
        return tuple(out)

    @cached_property
    def operands(self):
        # The edges of rule 5, without the opaque arguments.
        return self.operands_of(n for n in self.argument_names if n not in self.opaque)

    def bound(self):
        return []


class AbstractExpression(Base):
    # Every concrete class states its static type, with the helpers below where they apply.
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

    @cached_property
    def expression_operands(self):
        return tuple(x for x in self.operands if isinstance(x, AbstractExpression))

    def free_indices_of_operands(self):
        # The free indices of the operands, each Index once, in order of first occurrence.
        pairs = (p for x in self.expression_operands for p in x.free_indices)
        return tuple(dict(pairs).items())

    def support_of_operands(self):
        # The last support among the operands, or None.
        supports = (x.support for x in reversed(self.expression_operands))
        return next((s for s in supports if s is not None), None)


@final
class Index(Base):
    identifier = "uflx.Index"
    label: str | tuple
    opaque = ("label",)
    labels = ("label",)
