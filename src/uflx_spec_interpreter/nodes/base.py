from __future__ import annotations

import hashlib
import numbers
import weakref
from abc import ABCMeta
from functools import cached_property
from typing import ClassVar, dataclass_transform, final


@dataclass_transform(frozen_default=True, eq_default=False)
class Interned(ABCMeta):
    # Every node by its digest, so that equal nodes are the same object.
    _table: ClassVar[weakref.WeakValueDictionary] = weakref.WeakValueDictionary()

    def __new__(mcls, name, bases, namespace, **kwargs):
        # The construction arguments are the annotated names with no value, in order.
        if "__annotate__" in namespace and "__annotations__" not in namespace:
            raise TypeError(f"{name}: a node module needs from __future__ import annotations")
        own = [n for n in namespace.get("__annotations__", {}) if n not in namespace]
        if own:
            namespace["argument_names"] = tuple(own)
            namespace["__match_args__"] = tuple(own)
            for k, n in enumerate(own):
                namespace[n] = property(lambda self, k=k: self.construction_arguments[k])
        return super().__new__(mcls, name, bases, namespace, **kwargs)

    def __call__(cls, *args):
        if "identifier" not in cls.__dict__:
            raise TypeError(f"{cls.__name__} states no identifier")
        if len(args) != len(cls.argument_names):
            raise TypeError(f"{cls.__name__} takes the arguments {cls.argument_names}")
        digest = Interned.digest_of(cls.identifier, args)
        existing = Interned._table.get(digest)
        if existing is not None:
            return existing
        node = object.__new__(cls)
        object.__setattr__(node, "construction_arguments", args)
        object.__setattr__(node, "digest", digest)
        Interned._table[digest] = node
        return node

    @staticmethod
    def encode(value):
        # A canonical text, so 1, 1.0 and True differ, and so do 0.0 and -0.0.
        match value:
            case Base():
                return f"#{value.digest:032x}"
            case tuple():
                return "(" + ",".join(map(Interned.encode, value)) + ")"
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
        data = Interned.encode((identifier, tuple(values))).encode()
        return int.from_bytes(hashlib.blake2b(data, digest_size=16).digest(), "big")


class Base(metaclass=Interned):
    # The names of the construction arguments, which the metaclass reads from the annotations.
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

    @cached_property
    def operands(self):
        # The nodes among the construction arguments, through tuples, in order, as rule 5 says.
        out = []

        def walk(x):
            if isinstance(x, Base):
                out.append(x)
            elif isinstance(x, tuple):
                for y in x:
                    walk(y)

        for name, value in zip(self.argument_names, self.construction_arguments):
            if name not in self.opaque:
                walk(value)
        return tuple(out)

    def bound(self):
        return []


class AbstractExpression(Base):
    # The static type, derived once for each interned node. The defaults hold for most classes.
    @cached_property
    def expression_operands(self):
        return tuple(x for x in self.operands if isinstance(x, AbstractExpression))

    @cached_property
    def value_shape(self):
        return ()

    @cached_property
    def free_indices(self):
        out = {}
        for x in self.expression_operands:
            out |= dict(x.free_indices)
        return tuple(out.items())

    @cached_property
    def arity(self):
        operands = self.expression_operands
        return operands[0].arity if operands else 0

    @cached_property
    def support(self):
        found = None
        for x in self.expression_operands:
            if x.support is not None:
                found = x.support
        return found


@final
class Index(Base):
    identifier = "uflx.Index"
    label: str | tuple
    opaque = ("label",)
    labels = ("label",)
