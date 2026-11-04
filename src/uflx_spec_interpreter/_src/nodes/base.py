from __future__ import annotations

import inspect
from abc import ABC, abstractmethod, update_abstractmethods
from typing import TYPE_CHECKING, ClassVar, Literal, Never, Self, cast, dataclass_transform

from ..interning import _digest_of, _table
from ..tuples import _members
from ..types import _Construction

if TYPE_CHECKING:
    from ..nodes.domains import AbstractDomain
    from ..nodes.indices import AbstractExtent, GatheredIndex
    from ..nodes.spaces import AbstractFunctionSpace
    from ..nodes.topology import AbstractTopology
    from .forms import Function
    from .indices import Index


@dataclass_transform(frozen_default=True, eq_default=False)
class Base(ABC):
    # The names of the construction arguments, read from the annotations.
    argument_names: ClassVar[tuple[str, ...]] = ()
    # Expression operands evaluated by passes; all construction arguments remain dependencies.
    value_fields: ClassVar[tuple[str, ...]] = ()
    is_terminal: ClassVar[bool] = False
    # Index entries selected when instantiating a value, excluding identity labels.
    index_fields: ClassVar[tuple[str, ...]] = ()
    # Arguments whose nodes identify the node, which a substitution rebuilds.
    labels: ClassVar[tuple[str, ...]] = ()
    # Additional linear operands declared by extensions; core rules live in rules/.
    commuting: ClassVar[tuple[str, ...]] = ()
    # A binding node names the arguments where its bound names are in scope.
    scope: ClassVar[tuple[str, ...]] = ()

    identifier: ClassVar[str]
    __match_args__: ClassVar[tuple[str, ...]] = ()

    @property
    def construction_arguments(self) -> tuple[_Construction, ...]:
        return cast("tuple[_Construction, ...]", vars(self)["construction_arguments"])

    @property
    def digest(self) -> int:
        return cast("int", vars(self)["digest"])

    def __init_subclass__(cls, **kwargs: object) -> None:
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
        if not names:
            return
        cls.argument_names = cls.__match_args__ = names
        for position, name in enumerate(names):
            setattr(cls, name, _argument_property(position))
        # A generated property may implement an abstract property.
        update_abstractmethods(cls)

    def __new__(cls, *args: _Construction) -> Self:
        if "identifier" not in cls.__dict__:
            raise TypeError(f"{cls.__name__} states no identifier")
        if len(args) != len(cls.argument_names):
            raise TypeError(f"{cls.__name__} takes the arguments {cls.argument_names}")
        from ..validation import _construction_arguments

        _construction_arguments(args)
        digest = _digest_of(cls.identifier, args)
        key = (cls, digest)
        node = _table.get(key)
        if node is None:
            node = super().__new__(cls)
            vars(node).update(construction_arguments=args, digest=digest)
            node._validate()
            node._validate_type()
            _table[key] = node
        return cast("Self", node)

    def __setattr__(self, name: str, value: object) -> Never:
        raise AttributeError(f"{type(self).__name__} is immutable")

    def __hash__(self) -> int:
        return self.digest

    def __repr__(self) -> str:
        return f"{type(self).__name__}({', '.join(map(repr, self.construction_arguments))})"

    def _validate(self) -> None:
        pass

    def _validate_type(self) -> None:
        pass

    def bound(self) -> tuple[Index | Function, ...]:
        return ()


class AbstractExpression(Base):
    def _validate_type(self) -> None:
        from ..traversal import _expressions
        from ..validation import _merge_indices, _require

        _require(not self.is_terminal or not self.value_fields, "a terminal has no value operands")
        operands = _expressions(self)
        _merge_indices(*(x.free_indices for x in operands))
        _require(self.scalar_type in ("real", "complex"), "unknown scalar type")
        if self.matching_domains:
            domains = [x.domain for x in operands if x.domain is not None]
            _require(not domains or all(D == domains[0] for D in domains), "operand domains differ")

    matching_domains: ClassVar[bool] = False

    # Every concrete class states its static type, so the abstract class has no defaults.
    @property
    @abstractmethod
    def value_shape(self) -> tuple[int | AbstractExtent, ...]: ...

    @property
    @abstractmethod
    def free_indices(self) -> tuple[tuple[Index, int | AbstractExtent], ...]: ...

    @property
    @abstractmethod
    def argument_spaces(
        self,
    ) -> tuple[AbstractFunctionSpace | tuple[AbstractFunctionSpace, ...], ...]: ...

    @property
    @abstractmethod
    def domain(self) -> AbstractDomain | AbstractTopology | None: ...

    @property
    @abstractmethod
    def scalar_type(self) -> Literal["real", "complex"]: ...

    @staticmethod
    def joined_scalar_type(
        *operands: AbstractExpression | tuple[AbstractExpression, ...],
    ) -> Literal["real", "complex"]:
        # The default rule, complex if and only if an edge is a complex expression.
        from .scalar_types import ScalarType

        flat = [m for x in operands for m in _members(x)]
        return (
            ScalarType.complex
            if any(m.scalar_type == ScalarType.complex for m in flat)
            else ScalarType.real
        )

    # The free functions import the nodes, so each overload imports its free function locally.
    def __add__(self, other: object) -> AbstractExpression:
        from ..arithmetic import add

        if not isinstance(other, AbstractExpression):
            return NotImplemented
        return add(self, other)

    def __sub__(self, other: object) -> AbstractExpression:
        from ..arithmetic import subtract

        if not isinstance(other, AbstractExpression):
            return NotImplemented
        return subtract(self, other)

    def __mul__(self, other: object) -> AbstractExpression:
        from ..arithmetic import multiply

        if not isinstance(other, AbstractExpression):
            return NotImplemented
        return multiply(self, other)

    def __truediv__(self, other: object) -> AbstractExpression:
        from ..arithmetic import divide

        if not isinstance(other, AbstractExpression):
            return NotImplemented
        return divide(self, other)

    def __pow__(self, other: object) -> AbstractExpression:
        from ..arithmetic import power

        if not isinstance(other, AbstractExpression):
            return NotImplemented
        return power(self, other)

    def __matmul__(self, other: object) -> AbstractExpression:
        from ..tensors import matrix_product

        if not isinstance(other, AbstractExpression):
            return NotImplemented
        return matrix_product(self, other)

    def __neg__(self) -> AbstractExpression:
        from ..arithmetic import negate

        return negate(self)

    def __abs__(self) -> AbstractExpression:
        from ..arithmetic import absolute

        return absolute(self)

    def __getitem__(
        self, entries: int | Index | GatheredIndex | tuple[int | Index | GatheredIndex, ...]
    ) -> AbstractExpression:
        from ..tensors import component

        return component(self, _members(entries))

    def __iter__(self) -> Never:
        raise TypeError(f"{type(self).__name__} is not iterable")


def _argument_property(position: int) -> property:
    def read(node: Base) -> _Construction:
        return node.construction_arguments[position]

    return property(read)
