Software Organization
=====================
Cool tree cartoon of main files:

.. code-block:: text

	spacejam
	├── LICENSE.txt
	├── MANIFEST.in¶
	├── README.md
	├── dist
	│   ├── spacejam-0.0.3-py3-none-any.whl
	│   └── spacejam-0.0.3.tar.gz
	├── docs
	│   ├── Makefile
	│   └── figs
	│       ├── test.png
	│       └── test_ii.png
	├── requirements.txt
	├── setup.cfg
	├── setup.py
	└── spacejam
	    ├── __init__.py
	    ├── autodiff.py
	    ├── dual.py
	    ├── integrators.py
	    └── test
		├── __init__.py
		├── test_autodiff.py
		└── test_dual.py

Overview of main modules
------------------------
* ``autodiff.py``: Performs automatic differentiation of user-specified
  functions by following dual number rules provided by ``dual.py``

* ``dual.py``: Overloads basic math operations and returns an 
  automatic differentiation ``spacejam`` object

* ``integrators.py``: Suite of implicit integration schemes

* ``test_autodiff.py``: Test harness for class methods in ``autodiff.py``

* ``test_dual.py``: Test harness for class methods in ``dual.py``

* ``test_integrators.py``: coming soon

.. _install:

Installation
------------
Virtual Environment
~~~~~~~~~~~~~~~~~~~
For development, or just to have a self contained enviroment to use ``spacejam``
in, run the following commands anywhere on your computer:

.. code-block:: none                                                                                   
                                                                                    
        python -m venv venv                                                         
        source venv/bin/activate                                                    
        pip install spacejam

Tests
~~~~~
Unit tests are stored in ``spacejam/tests`` and each module mentioned above
has its own doctests. TravisCI and Coveralls integration is also provided. You
can run these tests and coverage reports yourself by doing the following:

.. code-block:: none

        cd venv/lib/python3.7/site-packages/spacejam
        pytest --doctest-modules --cov=. --cov-report term-missing

Check out :ref:`howto` for a quickstart tutorial.
