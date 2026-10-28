import hashlib
import numbers
import weakref
from dataclasses import dataclass, fields
from functools import cached_property

# Every node by its digest, so that equal nodes are the same object.
TABLE = weakref.WeakValueDictionary()


def encode(value):
    # A canonical encoding, tagged by type, so 1, 1.0 and True differ, and so do 0.0 and -0.0.
    match value:
        case Base():
            return b"N" + value.digest.to_bytes(16, "big")
        case None:
            return b"n"
        case bool():
            return b"b1" if value else b"b0"
        case numbers.Integral():
            return b"i" + str(int(value)).encode() + b";"
        case numbers.Real():
            return b"f" + float(value).hex().encode() + b";"
        case numbers.Complex():
            return b"c" + encode(float(value.real)) + encode(float(value.imag))
        case str():
            data = value.encode()
            return b"s" + str(len(data)).encode() + b":" + data
        case tuple():
            return b"t" + str(len(value)).encode() + b":" + b"".join(map(encode, value))
    raise TypeError(f"not a construction argument: {value!r}")


def digest_of(identifier, values):
    # Seed-free, unlike the built-in hash of str and bytes, so equal in every process.
    data = encode(identifier) + encode(tuple(values))
    return int.from_bytes(hashlib.blake2b(data, digest_size=16).digest(), "big")


class Interned(type):
    def __call__(cls, *args, **kwargs):
        if "identifier" not in cls.__dict__:
            raise TypeError(f"{cls.__name__} states no identifier")
        node = super().__call__(*args, **kwargs)
        digest = digest_of(cls.identifier, (getattr(node, f.name) for f in fields(node)))
        existing = TABLE.get(digest)
        if existing is not None:
            return existing
        object.__setattr__(node, "digest", digest)
        TABLE[digest] = node
        return node


class Base(metaclass=Interned):
    # Fields that are data, not operands, so traversal skips them.
    opaque: tuple = ()
    # Fields whose nodes identify the node, which a substitution rebuilds.
    labels: tuple = ()
    # A binding node names the fields where its bound names are in scope.
    scope: tuple = ()

    def bound(self):
        return []

    def __hash__(self):
        return self.digest % (2**61 - 1)


class AbstractExpression(Base):
    # The static type, derived once for each interned node. The defaults hold for most classes.
    def expression_operands(self):
        out = []

        def walk(x):
            if isinstance(x, AbstractExpression):
                out.append(x)
            elif isinstance(x, tuple):
                for y in x:
                    walk(y)

        for f in fields(self):
            if f.name not in self.opaque:
                walk(getattr(self, f.name))
        return out

    @cached_property
    def value_shape(self):
        return ()

    @cached_property
    def free_indices(self):
        out = {}
        for x in self.expression_operands():
            out |= dict(x.free_indices)
        return tuple(out.items())

    @cached_property
    def arity(self):
        operands = self.expression_operands()
        return operands[0].arity if operands else 0

    @cached_property
    def support(self):
        found = None
        for x in self.expression_operands():
            if x.support is not None:
                found = x.support
        return found


@dataclass(frozen=True, eq=False)
class Index(Base):
    identifier = "uflx.Index"
    label: object
    opaque = ("label",)
    labels = ("label",)
