Python API
==========

The core builds and rewrites symbolic expressions. The Basix package supplies
finite elements and quadrature, and the JAX compiler turns expressions into
callable kernels. See the :doc:`internal node reference <nodes>` for the
classes used to represent expressions.

Core language
-------------

``uflx.arithmetic``
~~~~~~~~~~~~~~~~~~~

.. automodule:: uflx.arithmetic
   :members:
   :undoc-members:

``uflx.calculus``
~~~~~~~~~~~~~~~~~

.. automodule:: uflx.calculus
   :members:
   :undoc-members:

``uflx.forms``
~~~~~~~~~~~~~~

.. automodule:: uflx.forms
   :members:
   :undoc-members:

``uflx.geometry``
~~~~~~~~~~~~~~~~~

.. automodule:: uflx.geometry
   :members:
   :undoc-members:

``uflx.tensors``
~~~~~~~~~~~~~~~~

.. automodule:: uflx.tensors
   :members:
   :undoc-members:

Rewriting and lowering
----------------------

``uflx.algorithms``
~~~~~~~~~~~~~~~~~~~

.. automodule:: uflx.algorithms
   :members:
   :undoc-members:

``uflx.passes``
~~~~~~~~~~~~~~~

.. automodule:: uflx.passes
   :members:
   :undoc-members:

Basix integration
-----------------

``uflx_basix.cells``
~~~~~~~~~~~~~~~~~~~~

.. automodule:: uflx_basix.cells
   :members:
   :undoc-members:

``uflx_basix.spaces``
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: uflx_basix.spaces
   :members:
   :undoc-members:

``uflx_basix.facets``
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: uflx_basix.facets
   :members:
   :undoc-members:

``uflx_basix.interpolation``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: uflx_basix.interpolation
   :members:
   :undoc-members:

``uflx_basix.quadrature``
~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: uflx_basix.quadrature
   :members:
   :undoc-members:

JAX compiler
------------

``uflx_jax_compiler``
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: uflx_jax_compiler
   :members:
   :undoc-members:

Extensions
----------

``uflx_extensions.complex_differentiation``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: uflx_extensions.complex_differentiation
   :members: complex_derivative_parts
   :undoc-members:

``uflx_extensions.external_operators``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: uflx_extensions.external_operators
   :members: external_operator, apply_external_derivatives
   :undoc-members:

``uflx_extensions.index_tables``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: uflx_extensions.index_tables
   :members: index_table
   :undoc-members:
