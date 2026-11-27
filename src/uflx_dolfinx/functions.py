"""Symbolic spaces and functions backed by DOLFINx data."""

from math import prod
from typing import TYPE_CHECKING

import basix
import numpy as np
from dolfinx import fem

from uflx._src.forms import as_function_space
from uflx.forms import function as symbolic_function
from uflx.geometry import create_mesh, euclidean_space
from uflx.nodes import AbstractExpression, Base, MappedSpace
from uflx_basix.cells import create_topology
from uflx_basix.nodes import BasixSpace
from uflx_basix.spaces import function_space as basix_space
from uflx_extensions.runtime_arrays import runtime_array


class _MeshIdentity(Base):
    identifier = "uflx_dolfinx.MeshIdentity"
    native_id: int


def _mesh_identity(mesh):
    native = mesh._cpp_object
    identity = _MeshIdentity(id(native))
    # Keep the native wrapper alive while any symbol refers to this identity.
    vars(identity).setdefault("mesh", native)
    return identity


def _physical_mesh(mesh):
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

    if TYPE_CHECKING:
        function_space: fem.FunctionSpace


@as_function_space.register(fem.FunctionSpace)
def functionspace(V):
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
    reference = _ReferenceSpace(*mapped.reference_space.construction_arguments, id(V._cpp_object))
    vars(reference).setdefault("function_space", V)
    return MappedSpace(reference, mapped.mesh, mapped.law)


class _TerminalMeta(type(AbstractExpression)):
    def __call__(cls, *args, **kwargs):
        # UFLx rebuilds an interned terminal from its symbolic operand.
        if len(args) == 1 and isinstance(args[0], AbstractExpression) and not kwargs:
            return super().__call__(*args)
        return cls._from_dolfinx(*args, **kwargs)


class _Terminal(AbstractExpression, metaclass=_TerminalMeta):
    is_terminal = True
    symbol: AbstractExpression

    @property
    def value_shape(self):
        return self.symbol.value_shape

    @property
    def domain(self):
        return self.symbol.domain

    @property
    def scalar_type(self):
        return self.symbol.scalar_type

    @property
    def argument_spaces(self):
        return ()

    @property
    def free_indices(self):
        return ()

    # Runtime storage is excluded from the immutable symbolic construction arguments.
    if TYPE_CHECKING:
        data: fem.Function | fem.Constant


class Function(_Terminal):
    """A live DOLFINx field usable as a UFLx expression."""

    identifier = "uflx_dolfinx.Function"

    @property
    def space(self):
        return self.symbol.space

    @property
    def x(self):
        return self.data.x

    def interpolate(self, *args, **kwargs):
        return self.data.interpolate(*args, **kwargs)

    @classmethod
    def _from_dolfinx(cls, V, name):
        """Return the live field identified by its symbolic space and name."""
        symbol = symbolic_function(functionspace(V), ("dolfinx.function", name))
        field = cls(symbol)
        if "data" not in vars(field):
            vars(field)["data"] = fem.Function(V, name=name, dtype=V.mesh.geometry.x.dtype)
        elif field.data.function_space._cpp_object != V._cpp_object:
            raise ValueError(f"{name!r} is already bound to a different DOLFINx function space")
        return field


class Constant(_Terminal):
    """A real DOLFINx constant with mutable value storage."""

    identifier = "uflx_dolfinx.Constant"

    @property
    def value(self):
        return self.data.value

    @classmethod
    def _from_dolfinx(cls, domain, c, *, name: str):
        """Return a named live constant. The value initializes its first declaration."""
        values = np.asarray(c)
        if values.dtype != domain.geometry.x.dtype:
            raise ValueError(
                f"constant has dtype {values.dtype}, expected {domain.geometry.x.dtype}"
            )
        symbol = runtime_array(("dolfinx.constant", _mesh_identity(domain), name), values.shape)
        field = cls(symbol)
        if "data" not in vars(field):
            vars(field)["data"] = fem.Constant(domain, values)
        return field
