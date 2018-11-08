Software Organization
=====================

Cool tree cartoon:

.. code-block:: text

   ├── LICENSE.rtf
   ├── MANIFEST.in¶
   ├── README.md
   ├── docs
   │   ├── Documentation.ipynb
   │   ├── Makefile
   │   ├── __init__.py
   │   └── milestone2.ipynb
   ├── requirements.txt
   ├── setup.cfg
   ├── setup.py
   └── spacejam
       ├── Dual.py
       ├── Jacobian.py
       ├── __init__.py
       └── tests
           ├── Demo.ipynb
           ├── DualNumbers_test.py
           └── init.py 


Overview of main modules
------------------------
* ``Dual.py``: Overloads basic math operations and returns an 
  automatic differentiation `spacejam` object

* ``DualNumbers_test.py``: Test harness for class methods in ``Dual.py``

Tests
-----
Unit tests are stored in ``spacejam/tests/DualNumbers.py`` and each
method in ``spacejam/Dual.py`` have their own doctests. `spacejam` also has
TravisCI and Coveralls integration.

Installation
------------
Until ``spacejam`` is made available on PyPI, the easiest way to install is to
run everything in a virtual environment.

First download the ``spacejam`` repo anywhere onto your computer with

.. code-block:: none 

        git clone git@github.com:cs207-SpaceJam/cs207-FinalProject.git

A virtual environment can be created by using the following
command:                               
                                                                                    
.. code-block:: none                                                                                   
                                                                                    
        python -m venv venv                                                         
                                                                                    
Next activate the environment by doing:                                                 
                                                                                    
.. code-block:: none
   
           source venv/bin/activate                                                    
                                                                                    
Finally, navigate into the root directory (cs207-FinalProject) and install the 
required dependencies (along with `spacejam``).

.. code-block:: none

        pip install -r requirements.txt

* Check out :ref:`howto` for a quick tutorial on getting started.
