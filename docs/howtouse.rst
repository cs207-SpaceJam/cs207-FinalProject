How to use ``spacejam``
========================

To install our package, first create a virtual environment using the following command (this should be done in the SpaceJam root folder):

::
        
        python -m venv venv

And proceed to activate it as such:

::

        source venv/bin/activate

If you run this in the root folder, you should also install all 
dependencies via:

::

        pip install -r requirements.txt

Now you should be able to proceed with the demo below. These following series 
of demos will walk you through how to differentiate a wide variety of 
functions with ``spacejam``

Demo I: Scalar function, scalar input
-------------------------------------
This is the simplest case, where the function you provide takes in a single 
scalar argument :math:`(x=a)` and outputs a single scalar value :math:`f(a)`.

For example, let us take a look at the function :math:`f(x) = x^3`, which you can define below as:

::

        def f(x):
            return x_1**3 + x_2

``spacejam`` just needs a value :math:`(x=p)` to evalute your function at and 
a dual number object :math:`p_x = f(x) + \epsilon_x` which is needed to perform 
the automatic differentiation of your function wrt. :math:`x`. Luckily, 
``spacejam`` creates this object for you with

::      

        import spacejam.Dual as d
        p = 5  # evaluation point
        p_x = d.Dual(p) # call Dual object

for an example value of :math:`x = 5` to evaluate your function at. Now, 
evaluating your function at :math:`x=5` and simultaneously computing the 
derivative at this point is as easy as

::
        
        ad = f(p_x)
        >>> ad
        125.00 + eps 75.00

where the real part is :math:`f(x=5) = 125` and the dual part is 
:math:`\left.\frac{\mathrm d f}{\mathrm d x}\right|_{x=5} = 75` .

The real and dual parts are also conveniently stored as attributes in the 
``spacejam`` object ``ad``,

::

        >>> print(ad.r) # real part f(x=p)
        125.0
        >>> print(ad.d) # dual part df/dx|x=p
        75.0

Note: The dual part is returned as a ``numpy`` array because 
``spacejam`` can also operate on multivariable functions and parameters, 
which we outline in `Demo II: Scalar function with vector input`_.
and `Demo III: Vector function with vector input`_.

Demo II: Scalar function with vector input
------------------------------------------
This next demo explores the case where a new example function $f$ can accept 
vector input, for example :math:`\mathbf p = (x_1, x_2) = (5, 2)` and return a 
single scalar value :math:`f(\mathbf p) = f(x_1, x_2) = 3x_1x_2 - 2x_2^3/x_1` .

The dual number objects are created in much the same way as in 
`Demo I <Demo I: Scalar function, scalar input_>`__,
with the only difference being the specification of separate dual number 
objects 

.. math::

        \begin{align*}
        p_{x_1} &= f(x_1, x_2) + \epsilon_{x_1} \frac{\partial f}{\partial x_1}
        - \epsilon_{x_2} 0\\
        p_{x_2} &= f(x_1, x_2) + \epsilon_{x_1} 0
        - \epsilon_{x_2} \frac{\partial f}{\partial x_2}
        \end{align*}\quad.

This is accomplished with the ``idx`` and ``x`` argument that you supply to
``spacejam`` so that it knows which dual parts need to be set to zero in the 
modified dual numbers above. In this modified setup, ``spacejam`` now returns

.. math::

        \begin{align*}
        f(\mathbf p) + \epsilon_{x_1}\frac{\partial f}{\partial x_1} 
        - \epsilon_{x_2}\frac{\partial f}{\partial x_2}
        \equiv f(\mathbf p) + \epsilon \left[\frac{\partial f}{\partial x_1}, 
        \frac{\partial f}{\partial x_2}\right] = f(\mathbf p) + \epsilon\nabla f
        \end{align*}\quad.

Applying this to the new function :math:`f` would look like the following

::

        import numpy as np 

        def f(x_1, x_2): 
        return 3*x_1*x_2 - 2*x_2**3/x_1

        p = np.array([5, 2]) # evaluation point (x_1, x_2) = (5, 2)

        p_x1 = d.Dual(p[0], idx=0, x=p) 
        p_x2 = d.Dual(p[1], idx=1, x=p)

        # print f(p) and grad(f) evaluated at p
        ad = f(p_x1, p_x2)
        
        >>> print(ad)
        26.80 + eps [ 6.64 10.2 ]

The real and dual parts can again be accessed with

::

        >>> print(ad.r)
        26.8
        >>> print(ad.d)
        [ 6.64 10.2 ]

Demo III: Vector function with vector input
-------------------------------------------
This final demo shows how to use ``spacejam`` to simultaneously evaluate the
example vector function

.. math::

        \mathbf{F} = \begin{bmatrix}f_1(x_1, x_2)\\f_2(x_1, x_2)
        \\f_{3}(x_1, x_2)\end{bmatrix}
        = \begin{bmatrix}
        x_1^2 + x_1x_2 + 2 \\ x_1x_2^3 + x_1^2 \\ x_2^3/x_1 + x_1 + x_1^2x_2^2 + x_2^4
        \end{bmatrix}

and its Jacobian,

.. math::

        \mathbf J = \begin{bmatrix}
        \nabla f_1(x_1, x_2) \\ \nabla f_2(x_1, x_2) \\ \nabla f_3(x_1, x_2)
        \end{bmatrix}\quad.

at the point :math:`\mathbf{p} = (x_1, x_2) = (1, 2)` .

The configuration of ``spacejam`` happens to be exactly the same as in 
`Demo II <Demo II: Scalar function with vector input_>`__, and would look like 
the following

::

        def F(x, y):
        f1 = x**2 + x*y + 2
        f2 = x*y**3 + x**2
        f3 = y**3/x + x + x**2*y**2 + y**4
        return np.array([f1, f2, f3])

        p = np.array([1, 2])
        p_x = d.Dual(p[0], idx=0, x=p)
        p_y = d.Dual(p[1], idx=1, x=p)

        ad = F(p_x, p_y)
        
        >>> print(ad)
        [5.00 + eps [4. 1.], 9.00 + eps [10. 12.], 29.00 + eps [ 1. 48.]]

For each :math:`i` th entry, in the 1D ``numpy`` array `ad`, the real part is 
the :math:`i` th component of :math:`\mathbf{F}(\mathbf{p})` and the dual 
part is the corresponding row in the Jacobian :math:`\mathbf J` evaluated at 
:math:`\mathbf p = (x_1, x_2) = (1, 2)` .

The output can be cleaned up a bit to shape :math:`\mathbf J` into its matrix 
form ``Jac`` with,

::

        Jac = np.empty((F(*p).size, p.size))
        for i, f in enumerate(ad):
            Jac[i] = f.d
        
        >>> print(Jac)
        [[ 4.,  1.],
        [10., 12.],
        [ 1., 48.]]
