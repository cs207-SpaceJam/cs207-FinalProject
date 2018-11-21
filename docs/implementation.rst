.. _api:

Implementation Details
======================

Data Structures
---------------
* ``spacejam`` uses 1D ``numpy.ndarrays`` to return partial derivatives, where the
  :math:`j` th entry contains :math:`\frac{\partial f_i}{\partial x_j}` for :math:`i =
  1, ... m` and :math:`j = 1, ... k ` for :math:`m` different functions that are a function of :math:`k` different variables.

  TODO:
* The convenience function <insert here> stacks these 1D arrays into 
  an :math:`m\times k` ``numpy.ndarray`` Jacobian matrix for ease of viewing.
  
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

