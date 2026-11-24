Internal node reference
=======================

The interpreter represents expressions as graphs of immutable nodes. The
public :doc:`Python API <api>` builds those graphs. Use this reference when
inspecting an expression, writing a rewrite rule or implementing a language
extension. For example, :func:`uflx.forms.form` creates a
:class:`~uflx.nodes.Form` node whose fields hold the expression and its
argument bindings. Concrete node signatures list the construction fields in
positional order.

Core nodes
----------

.. automodule:: uflx.nodes
   :members:
   :undoc-members:
   :exclude-members: ExpressionScalarType, ScalarType

.. autoclass:: uflx._src.nodes.expressions.BindingExpression
   :members:

Basix nodes
-----------

.. automodule:: uflx_basix.nodes
   :members:
   :undoc-members:
   :exclude-members: table

.. autoclass:: uflx_basix._elements.BuiltinElement

.. autoclass:: uflx_basix._elements.CustomElement

Extension nodes
---------------

.. autoclass:: uflx_extensions.external_operators.AbstractOperation
   :members:

.. autoclass:: uflx_extensions.external_operators.ExternalOperator
   :members:

.. autoclass:: uflx_extensions.index_tables.IndexArray
   :members:

.. autoclass:: uflx_extensions.index_tables.IndexTable
   :members:

.. autoclass:: uflx_extensions.runtime_arrays.RuntimeArray
   :members:

.. autoclass:: uflx_extensions.runtime_arrays.RuntimeExtent
   :members:
