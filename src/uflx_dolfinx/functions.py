"""Symbolic spaces and functions backed by DOLFINx data."""

from __future__ import annotations

from collections.abc import Callable
from math import prod
from typing import TYPE_CHECKING, Any, Self, TypeVar, cast

import basix
import numpy as np
import numpy.typing as npt
from dolfinx import fem, la
from dolfinx.mesh import Mesh as NativeMesh

from uflx._src.forms import as_function_space
from uflx.forms import function as symbolic_function
from uflx.geometry import create_mesh, euclidean_space
from uflx.nodes import (
    AbstractDomain,
    AbstractExpression,
    AbstractFunctionSpace,
    AbstractTopology,
    Base,
    MappedSpace,
    Mesh,
)
from uflx.nodes import Function as SymbolicFunction
from uflx.typing import ExpressionScalarType, Extent
from uflx_basix.cells import create_topology
from uflx_basix.nodes import BasixSpace
from uflx_basix.spaces import function_space as basix_space
from uflx_extensions.runtime_arrays import RuntimeArray, runtime_array


class _MeshIdentity(Base):
    identifier = "uflx_dolfinx.MeshIdentity"
    native_id: int


def _mesh_identity(mesh: NativeMesh) -> _MeshIdentity:
    native = mesh._cpp_object
    identity = _MeshIdentity(id(native))
    # Keep the native wrapper alive while any symbol refers to this identity.
    vars(identity).setdefault("mesh", native)
    return identity


def _physical_mesh(mesh: NativeMesh) -> Mesh:
    cell = mesh.basix_cell()
    (cmap,) = mesh.geometry.cmaps
    element = basix.create_element(
        basix.ElementFamily.P,
        cell,
        cmap.degree,
        basix.LagrangeVariant(cmap.variant),
        discontinuous=cmap.is_discontinuous,
        dtype=cmap.dtype,
    )
    if element.hash() != cmap.hash():
        raise ValueError("cannot reconstruct the mesh coordinate element from its DOLFINx map")
    gdim = mesh.geometry.dim
    X = basix_space(create_topology(cell), element, shape=(gdim,))
    x = symbolic_function(X, ("dolfinx.mesh", _mesh_identity(mesh)))
    return create_mesh(x, euclidean_space(gdim))


class _ReferenceSpace(BasixSpace):
    """Reference element with the identity of its live DOLFINx function space."""

    identifier = "uflx_dolfinx.ReferenceSpace"
    native_id: int

    @property
    def function_space(self) -> fem.FunctionSpace:
        return vars(self)["function_space"]


@as_function_space.register(fem.FunctionSpace)
def functionspace(V: fem.FunctionSpace) -> MappedSpace:
    """Describe a function space on its native DOLFINx mesh."""
    element = V.element
    base = basix.finite_element.FiniteElement(element.basix_element)
    # A scalar Basix element can be repeated in a vector or tensor block.
    shape = tuple(map(int, element.value_shape)) if not base.value_shape else ()
    if prod(shape) * base.dim != element.space_dimension:
        raise ValueError("the DOLFINx element layout is incompatible with a full Basix block")
    mapped = basix_space(_physical_mesh(V.mesh), base, shape=shape)
    # Cloned native spaces can share a mesh, element and dofmap. Retaining V
    # also keeps its native wrapper alive for the lifetime of this identity.
    base_space = cast(BasixSpace, mapped.reference_space)
    reference = _ReferenceSpace(
        base_space.support,
        base_space.element_data,
        base_space._scalar_type,
        base_space.block_shape,
        id(V._cpp_object),
    )
    vars(reference).setdefault("function_space", V)
    return MappedSpace(reference, mapped.mesh, mapped.law)


_T = TypeVar("_T", bound="_Terminal")


class _TerminalMeta(type(AbstractExpression)):
    def __call__(cls: type[_T], *args: Any, **kwargs: Any) -> _T:
        # UFLx rebuilds an interned terminal from its symbolic operand.
        if len(args) == 1 and isinstance(args[0], AbstractExpression) and not kwargs:
            return super().__call__(*args)  # pyrefly: ignore[bad-argument-type]
        return cls._from_dolfinx(*args, **kwargs)


class _Terminal(AbstractExpression, metaclass=_TerminalMeta):
    is_terminal = True
    symbol: AbstractExpression

    if TYPE_CHECKING:

        @classmethod
        def _from_dolfinx(cls, *args: Any, **kwargs: Any) -> Self: ...

    @property
    def value_shape(self) -> tuple[Extent, ...]:
        return self.symbol.value_shape

    @property
    def domain(self) -> AbstractDomain | AbstractTopology | None:
        return self.symbol.domain

    @property
    def scalar_type(self) -> ExpressionScalarType:
        return self.symbol.scalar_type

    @property
    def argument_spaces(self) -> tuple[()]:
        return ()

    @property
    def free_indices(self) -> tuple[()]:
        return ()


class Function(_Terminal):
    """A live DOLFINx field usable as a UFLx expression."""

    identifier = "uflx_dolfinx.Function"

    symbol: SymbolicFunction

    if TYPE_CHECKING:
        data: fem.Function

        def __init__(self, V: fem.FunctionSpace, name: str) -> None: ...

    @property
    def space(self) -> AbstractFunctionSpace:
        return self.symbol.space

    @property
    def x(self) -> la.Vector:
        return self.data.x

    def interpolate(
        self,
        u0: Callable[..., npt.NDArray] | fem.Expression | fem.Function,
        cells0: npt.NDArray[np.int32] | None = None,
        cells1: npt.NDArray[np.int32] | None = None,
    ) -> None:
        self.data.interpolate(u0, cells0, cells1)

    @classmethod
    def _from_dolfinx(cls, V: fem.FunctionSpace, name: str) -> Self:
        """Return the live field identified by its symbolic space and name."""
        symbol = symbolic_function(functionspace(V), ("dolfinx.function", name))
        field = cast(Callable[[AbstractExpression], Self], cls)(symbol)
        if "data" not in vars(field):
            vars(field)["data"] = fem.Function(V, name=name, dtype=V.mesh.geometry.x.dtype)
        elif field.data.function_space._cpp_object != V._cpp_object:
            raise ValueError(f"{name!r} is already bound to a different DOLFINx function space")
        return field


class Constant(_Terminal):
    """A real DOLFINx constant with mutable value storage."""

    identifier = "uflx_dolfinx.Constant"

    symbol: RuntimeArray

    if TYPE_CHECKING:
        data: fem.Constant

        def __init__(self, domain: NativeMesh, c: npt.ArrayLike, *, name: str) -> None: ...

    @property
    def value(self) -> npt.NDArray:
        return self.data.value

    @classmethod
    def _from_dolfinx(cls, domain: NativeMesh, c: npt.ArrayLike, *, name: str) -> Self:
        """Return a named live constant. The value initializes its first declaration."""
        values = np.asarray(c)
        if values.dtype != domain.geometry.x.dtype:
            raise ValueError(
                f"constant has dtype {values.dtype}, expected {domain.geometry.x.dtype}"
            )
        symbol = runtime_array(("dolfinx.constant", _mesh_identity(domain), name), values.shape)
        field = cast(Callable[[AbstractExpression], Self], cls)(symbol)
        if "data" not in vars(field):
            vars(field)["data"] = fem.Constant(domain, values)
        return field
