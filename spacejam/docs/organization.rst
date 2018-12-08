Software Organization
=====================
Cool tree cartoon of main files:

.. code-block:: text

        ├── README.md 
        ├── setup.py¶
        ├── readthedocs.yml
        ├── spacejam 
        │     └── _pychache_
   	      ├──build
	      ├──spacejam
		    └──_init_.py
		    └──_pycache_
		    └──autodiff.py
		    ├──dual.py
		    ├──intergrators.py
		    ├──test
			└──init.py
			└──test_autodiff.py
			└──test_dual.py
			└──test_integrators.py	
	      ├──spacejam.egg-info 
	      ├──docs
	      ├──LICENSE.txt
	      ├──MANIFEST.in
	      ├──README.md
	      ├──requirements.txt
	      ├──setup.cfg
	      ├──setup.py 
       
   
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
