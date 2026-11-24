Poisson on a triangle
=====================

Build a cell stiffness matrix and load vector for Poisson’s equation with a
unit source. The basis functions and quadrature live on the reference triangle.
The physical triangle is set by the vertices passed to the compiled kernels.

Run the example from the repository root with ``python examples/poisson.py``,
or :download:`download the script <../examples/poisson.py>`.

.. literalinclude:: ../examples/poisson.py
   :language: python
   :start-at: import basix
