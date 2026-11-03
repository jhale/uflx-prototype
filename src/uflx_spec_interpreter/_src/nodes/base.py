from __future__ import annotations

import inspect
from abc import ABC, abstractmethod, update_abstractmethods
from typing import ClassVar, dataclass_transform

from ..interning import digest_of, table
from ..tuples import members


@dataclass_transform(frozen_default=True, eq_default=False)
class Base(ABC):
    # The names of the construction arguments, read from the annotations.
    argument_names: ClassVar[tuple] = ()
    # Expression operands evaluated by passes; all construction arguments remain dependencies.
    value_fields: ClassVar[tuple] = ()
    is_terminal: ClassVar[bool] = False
    # Index entries selected when instantiating a value, excluding identity labels.
    index_fields: ClassVar[tuple] = ()
    # Arguments whose nodes identify the node, which a substitution rebuilds.
    labels: ClassVar[tuple] = ()
    # The operands into which a linear evaluation, such as a derivative or an action, moves.
    commuting: ClassVar[tuple] = ()
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
        from ..validation import construction_arguments

        construction_arguments(args)
        digest = digest_of(cls.identifier, args)
        key = (cls, digest)
        node = table.get(key)
        if node is None:
            node = super().__new__(cls)
            vars(node).update(construction_arguments=args, digest=digest)
            node._validate()
            node._validate_type()
            table[key] = node
        return node

    def __setattr__(self, name, value):
        raise AttributeError(f"{type(self).__name__} is immutable")

    def __hash__(self):
        return self.digest

    def __repr__(self):
        return f"{type(self).__name__}({', '.join(map(repr, self.construction_arguments))})"

    def _validate(self):
        pass

    def _validate_type(self):
        pass

    def bound(self):
        return []


class AbstractExpression(Base):
    def _validate_type(self):
        from ..substitution import expressions
        from ..validation import merge_indices, require

        require(not self.is_terminal or not self.value_fields, "a terminal has no value operands")
        operands = expressions(self)
        merge_indices(*(x.free_indices for x in operands))
        require(self.scalar_type in ("real", "complex"), "unknown scalar type")
        if self.matching_domains:
            domains = [x.domain for x in operands if x.domain is not None]
            require(not domains or all(D == domains[0] for D in domains), "operand domains differ")

    matching_domains: ClassVar[bool] = False

    # Every concrete class states its static type, so the abstract class has no defaults.
    @property
    @abstractmethod
    def value_shape(self) -> tuple: ...

    @property
    @abstractmethod
    def free_indices(self) -> tuple: ...

    @property
    @abstractmethod
    def argument_spaces(self) -> tuple: ...

    @property
    @abstractmethod
    def domain(self): ...

    @property
    @abstractmethod
    def scalar_type(self) -> str: ...

    @staticmethod
    def joined_scalar_type(*operands):
        # The default rule, complex if and only if an edge is a complex expression.
        from .scalar_types import ScalarType

        flat = [m for x in operands for m in members(x)]
        return (
            ScalarType.complex
            if any(m.scalar_type == ScalarType.complex for m in flat)
            else ScalarType.real
        )

    # The free functions import the nodes, so each overload imports its free function locally.
    def __add__(self, other):
        from ..arithmetic import add

        if not isinstance(other, AbstractExpression):
            return NotImplemented
        return add(self, other)

    def __sub__(self, other):
        from ..arithmetic import subtract

        if not isinstance(other, AbstractExpression):
            return NotImplemented
        return subtract(self, other)

    def __mul__(self, other):
        from ..arithmetic import multiply

        if not isinstance(other, AbstractExpression):
            return NotImplemented
        return multiply(self, other)

    def __truediv__(self, other):
        from ..arithmetic import divide

        if not isinstance(other, AbstractExpression):
            return NotImplemented
        return divide(self, other)

    def __pow__(self, other):
        from ..arithmetic import power

        if not isinstance(other, AbstractExpression):
            return NotImplemented
        return power(self, other)

    def __matmul__(self, other):
        from ..tensors import matrix_product

        if not isinstance(other, AbstractExpression):
            return NotImplemented
        return matrix_product(self, other)

    def __neg__(self):
        from ..arithmetic import negate

        return negate(self)

    def __abs__(self):
        from ..arithmetic import absolute

        return absolute(self)

    def __getitem__(self, entries):
        from ..tensors import component

        return component(self, members(entries))

    def __iter__(self):
        raise TypeError(f"{type(self).__name__} is not iterable")
