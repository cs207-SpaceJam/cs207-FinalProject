.. _api:

Implementation Details
======================

Data Structures
---------------
* ``spacejam`` uses 1D ``numpy.ndarrays`` to return partial derivatives, where the
  :math:`n` th entry contains :math:`\frac{\partial f_i}{\partial x_n}` for :math:`i =
  0, ... m` for :math:`m` different functions specified by the user.

  TODO:
* The convenience function <insert here> stacks these 1D arrays into 
  an :math:`m\times n` ``numpy.ndarray`` Jacobian matrix for ease of viewing.
  
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

