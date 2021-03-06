.. _api:

Implementation Details
======================

Data Structures
---------------
* ``spacejam`` uses 1D ``numpy.ndarrays`` to return partial derivatives, where
  the :math:`j` th entry contains :math:`\frac{\partial f_i}{\partial x_j}` for
  :math:`i = 1, ... m` and :math:`j = 1, ... k`. In general, this is for
  :math:`m` different functions that are a function of :math:`k` different
  variables.

* The internal convenience function :any:`spacejam.autodiff.AutoDiff._matrix`
  stacks these 1D arrays into an :math:`m\times k` ``numpy.ndarray`` Jacobian
  matrix for ease of viewing, as described in :ref:`diii`.
  
API
---
``spacejam.autodiff``
~~~~~~~~~~~~~~~~~~~~~
.. automodule:: spacejam.autodiff
   :members:

``spacejam.dual``
~~~~~~~~~~~~~~~~~
.. automodule:: spacejam.dual
   :members:

.. _integrators:

``spacejam.integrators``
~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: spacejam.integrators
   :members:
