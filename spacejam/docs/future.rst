Future
======

* Improve UI

  + Read in user-defined equations (e.g. json, yaml)

* Generalize algorithms

  + Add support for systems with time-dependent system of equations
  + Add IMEX schemes to integration suite 

* Add vector support

  + Make ``spacejam`` object indexable so you can do stuff like this:

::

        Z = sj.Dual([1, 2], [3, 4])

        print(z[0], z[1])

::


        1.00 + eps 3.00, 2.00 + eps 4.00 
