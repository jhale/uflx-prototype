# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT
"""Type annotations for UFLx programs and extensions.

``Argument[T]`` describes one argument with leaves of type ``T``.
``ArgumentSpaces`` describes all slots of a form, and ``FormEntry`` describes one
binding declaration. For example, ``((V, Q), M)`` is one argument space for
a velocity-pressure state and a control.

Use ``BindingEntry`` for a form entry or a bound index and ``Construction`` for a
node constructor argument. Scalar categories use ``ExpressionScalarType`` and
``NumericalScalarType``, with values named by ``ScalarType``. Indexing uses
``Extent`` and ``IndexEntry``, and transformation laws use ``Variance``.
Rewriting and lowering use ``Transform``, ``DerivativeRule`` and ``ContractViolation``.
"""

from ._src.nodes.scalar_types import ScalarType
from ._src.typing import Argument, ExpressionScalarType, NumericalScalarType, Variance
from ._src.typing.nodes import (
    ArgumentSpaces,
    BindingEntry,
    Construction,
    ContractViolation,
    DerivativeRule,
    Extent,
    FormEntry,
    IndexEntry,
    Transform,
)

__all__ = [
    "Argument",
    "ArgumentSpaces",
    "BindingEntry",
    "Construction",
    "ContractViolation",
    "DerivativeRule",
    "ExpressionScalarType",
    "Extent",
    "FormEntry",
    "IndexEntry",
    "NumericalScalarType",
    "ScalarType",
    "Transform",
    "Variance",
]
