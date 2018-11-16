.. _howto:

How to use ``spacejam``
========================
The following series of demos will step through how to differentiate a wide
variety of functions with ``spacejam`` .

Demo I: Scalar function, scalar input
-------------------------------------
This is the simplest case, where the function you provide takes in a single 
scalar argument :math:`(x=a)` and outputs a single scalar value :math:`f(a)`.

For example, let's take a look at the function :math:`f(x) = x^3`, which you
can define below as:

::

        import numpy as np

        def f(x):
            return np.array([x**3])

.. testsetup::

        import numpy as np

        def f(x):
            return np.array([x**3])

All ``spacejam`` needs now is for you to specify a point :math:`\mathbf p`
where you would like to evaluate your function at:

::

        p = np.array([5])  # evaluation point

.. testsetup::

        p = np.array([5])  # evaluation point

Now, evaluating your function and simultaneously computing the 
derivative with ``spacejam`` at this point is as easy as:

::

        import spacejam as sj

        ad = sj.AutoDiff(f, p)


.. testsetup::

        import spacejam as sj

        ad = sj.AutoDiff(f, p)

The real part of `ad` is now :math:`f(x=5) = 125` and the dual part is
:math:`\left.\frac{\mathrm d f}{\mathrm d x}\right|_{x=5} = 75` .

Theses real and dual parts are conveniently stored, respectively, as the ``r`` and ``d``
attributes in ``ad`` and can easily be printed to examine:

.. testcode::

        print(f'f(x) evaluated at p:\n{ad.r}\n\n'
              f'derivative of f(x) evaluated at p:\n{ad.d}')

.. testoutput::

        f(x) evaluated at p:
        [125.00]

        derivative of f(x) evaluated at p:
        [75.00]

Note: ``numpy`` arrays are used when defining your function and returning
results because ``spacejam`` can also operate on multivariable functions and
parameters, which we outline in `Demo II: Scalar function with vector input`_.
and `Demo III: Vector function with vector input`_.

Demo II: Scalar function with vector input
------------------------------------------
This next demo explores the case where a new example function :math:`f` can
accept vector input, for example :math:`\mathbf p = (x_1, x_2) = (5, 2)` and
return a single scalar value 
:math:`f(\mathbf p) = f(x_1, x_2) = 3x_1x_2 - 2x_2^3/x_1` 

The dual number objects are created in much the same way as in 
`Demo I <Demo I: Scalar function, scalar input_>`__, where:

.. math::

        \begin{align*}
        p_{x_1} &= f(x_1, x_2) + \epsilon_{x_1} \frac{\partial f}{\partial x_1}
        - \epsilon_{x_2} 0\\
        p_{x_2} &= f(x_1, x_2) + \epsilon_{x_1} 0
        - \epsilon_{x_2} \frac{\partial f}{\partial x_2}
        \end{align*}\quad,

as described in :ref:`ad`. Internally, this is accomplished with the ``idx``
and ``x`` argument in ``spacejam.dual`` so that it knows which dual parts need
to be set to zero in the modified dual numbers above. ``spacejam.autodiff``
then performs the following internally:

.. math::

        \begin{align*}
        f(\mathbf p) + \epsilon_{x_1}\frac{\partial f}{\partial x_1} 
        - \epsilon_{x_2}\frac{\partial f}{\partial x_2}
        \equiv f(\mathbf p) + \epsilon \left[\frac{\partial f}{\partial x_1}, 
        \frac{\partial f}{\partial x_2}\right] = f(\mathbf p) + \epsilon\nabla f
        \end{align*}\quad.

**tl;dr:** all that needs to be done is:

.. testcode::

        import numpy as np 
        import spacejam as sj

        def f(x_1, x_2): 
            return np.array([3*x_1*x_2 - 2*x_2**3/x_1])

        p = np.array([5, 2]) # evaluation point (x_1, x_2) = (5, 2)

        ad = sj.AutoDiff(f, p) # create spacejam object

        # check out the results
        print(f'f(x) evaluated p:\n{ad.r}\n\n'
              f'grad of f(x) evaluated at p:\n{ad.d}')

.. testoutput::

        f(x) evaluated p:
        [26.80]

        grad of f(x) evaluated at p:
        [6.64 10.20]

.. _diii:

Demo III: Vector function with vector input
-------------------------------------------
This final demo shows how to use ``spacejam`` to simultaneously evaluate the
example vector function:

.. math::

        \mathbf{F} = \begin{bmatrix}f_1(x_1, x_2)\\f_2(x_1, x_2)
        \\f_{3}(x_1, x_2)\end{bmatrix}
        = \begin{bmatrix}
        x_1^2 + x_1x_2 + 2 \\ x_1x_2^3 + x_1^2 \\ x_2^3/x_1 + x_1 + x_1^2x_2^2 + x_2^4
        \end{bmatrix}

and its Jacobian:

.. math::

        \mathbf J = \begin{bmatrix}
        \nabla f_1(x_1, x_2) \\ \nabla f_2(x_1, x_2) \\ \nabla f_3(x_1, x_2)
        \end{bmatrix}\quad.

at the point :math:`\mathbf{p} = (x_1, x_2) = (1, 2)` .

The interface with ``spacejam`` happens to be exactly the same as in the
previous two demos, only now your :math:`F(x)` will return a 1D ``numpy`` array
of functions :math:`(f_1, f_2, f_3)`:

.. testcode::

        # your (n) system of equations: 
        # F(x_1, x_2, ..., x_m) = (f1, f2, ..., f_n)
        def F(x_1, x_2):
                f_1 = x_1**2 + x_1*x_2 + 2
                f_2 = x_1*x_2**3 + x_1**2
                f_3 = x_1 + x_1**2*x_2**2 + x_2**3/x_1 + x_2**4
                return np.array([f_1, f_2, f_3])

        # where you want them evaluated at: 
        # p = (x_1, x_2, ..., x_m)
        p = np.array([1, 2])

        # auto differentiate!
        ad = sj.AutoDiff(F, p)

        # check out the results
        print(f'F(x) evaluated at p:\n{ad.r}\n\n'
              f'Jacobian of F(x) evaluated at p:\n{ad.d}')

.. testoutput::

        F(x) evaluated at p:
        [[5.00]
         [9.00]
         [29.00]]

        Jacobian of F(x) evaluated at p:
        [[4.00 1.00]
         [10.00 12.00]
         [1.00 48.00]]

Internally, for each :math:`i` th entry, in the 1D ``numpy`` array `ad._full`, the real part is 
the :math:`i` th component of :math:`\mathbf{F}(\mathbf{p})` and the dual 
part is the corresponding row in the Jacobian :math:`\mathbf J` evaluated at 
:math:`\mathbf p = (x_1, x_2) = (1, 2)` .

This is done in ``spacejam._jac`` for you with:

.. testcode::

       result = ad._full
       evals = np.empty((F(*p).size, 1)) # initialze empty F(p)
       jac = np.empty((F(*p).size, p.size)) # initialize empty J F(p)

       for i, f in enumerate(result): # fill in each row of each
           evals[i] = f.r
           jac[i] = f.d

       print(f'formated F(p):\n{evals}\n\nformated J F(p):\n{jac}') 


.. testoutput::

        formated F(p):
        [[5.00]
         [9.00]
         [29.00]]

        formated J F(p):
        [[4.00 1.00]
         [10.00 12.00]
         [1.00 48.00]]

where:

.. testcode::

        print(ad._full)

.. testoutput::

        [5.00 + eps [4.00 1.00] 9.00 + eps [10.00 12.00] 29.00 + eps [1.00 48.00]]

