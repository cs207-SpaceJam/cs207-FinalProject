Software Organization
=====================
Cool tree cartoon of main files:

.. code-block:: text

        ├── LICENSE.rtf
        ├── MANIFEST.in¶
        ├── README.md
        ├── docs
        │   ├── Makefile
        │   └── _images
        │       └── spring.png
        ├── readthedocs.yml
        ├── requirements.txt
        ├── setup.cfg
        ├── setup.py
        └── spacejam
            ├── __init__.py
            ├── autodiff.py
            ├── dual.py
            └── tests
                ├── dualnumbers_test.py
                └── init.py

Overview of main modules
------------------------
* ``Dual.py``: Overloads basic math operations and returns an 
  automatic differentiation ``spacejam`` object

* ``DualNumbers_test.py``: Test harness for class methods in ``Dual.py``

Tests
-----
Unit tests are stored in ``spacejam/tests/DualNumbers.py`` and each
method in ``spacejam/Dual.py`` have their own doctests. ``spacejam`` also has
TravisCI and Coveralls integration.

These tests can be run with the following command in the root directory:

.. code-block:: none

        pytest --doctest-modules --cov=. --cov-report term-missing

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


* Check out :ref:`howto` for a quick tutorial on what you can do.
