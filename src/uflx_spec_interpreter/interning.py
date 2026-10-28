import hashlib
import weakref

# Every node by its digest, so that equal nodes are the same object.
table = weakref.WeakValueDictionary()


def encode(value):
    # A canonical text: a node by its digest, a tuple by its items, anything else by type and repr.
    if isinstance(value, tuple):
        return "(" + ",".join(map(encode, value)) + ")"
    digest = getattr(value, "digest", None)
    return f"#{digest:032x}" if digest is not None else f"{type(value).__name__}:{value!r}"


def digest_of(identifier, values):
    # Seed-free, unlike the built-in hash of str and bytes, so equal in every process.
    data = encode((identifier, tuple(values))).encode()
    return int.from_bytes(hashlib.blake2b(data, digest_size=16).digest(), "big")
