UFLx prototype
==============

UFLx is an experimental language for finite element variational forms. This
prototype implements the core language, interfaces with Basix for finite
elements and quadrature, and uses JAX to compile cell kernels.
See the `UFLx core specification <https://fenics.github.io/uflx/language/>`_
for the language rules.

Installation
------------

To install the prototype, run these commands with Python 3.11 or later.

.. code-block:: bash

   git clone https://github.com/jhale/uflx-prototype.git
   cd uflx-prototype
   python -m pip install -e .

.. toctree::
   :maxdepth: 1

   poisson
   api
   nodes
