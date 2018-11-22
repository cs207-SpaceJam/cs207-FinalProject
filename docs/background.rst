Background
==========

.. _numerical:

Numerical Integration: A brief crash couse
------------------------------------------
Many physical systems can be expressed as a series of differential equations.
Euler's method is the simplest numerical procedure for solving these equations
given some initial conditions. In the case of our problem statement:

* We have some initial conditions (such as position and velocity of a planet)
  and we want to know what kind of orbit this planet will trace out, given that
  only the force acting on it is gravity. 
  
* Using the physical insight that
  the "slope" of position over time is velocity, and the "slope" of velocity
  over time is acceleration, we can predict, or integrate, how the quantities
  will change over time.  

* More explicitly, we can use the acceleration
  supplied by gravity to predict the velocity of our planet, and then use this
  velocity to predict its position a timestep :math:`\Delta t` later.  
  
* This gives us a new position and the whole process starts over again at the
  next timestep. Here is a schematic of the Euler integration method.

.. image:: http://jmahaffy.sdsu.edu/courses/f00/math122/lectures/num_method_diff_equations/images/euler_ani.gif

This plot above could represent the component of the planet's velocity varies
over time. Specifically, we have some solution curve (black) that we want to
approximate (red), given that we only know two things:

* where we started :math:`(t_0, y_0)`

* the rate of how where we were changes with time 
  :math:`\left(\dot{y}_0 \equiv \frac{\mathrm d y_0}{\mathrm{d} t}
  = \frac{y_1 - y_0}{h}\right)`

The cool thing about this is that even though we do not explicity know what
:math:`y_1` is, the fact that we are given :math:`\dot{y}_0` from the initial
conditions allows us to bootstrap our way around this. Starting with the
definition of slope, we can use the timestep :math:`h \equiv \Delta t = t_{n+1}
- t_n`, to find where we will be a timestep later :math:`\dot{y}_1`: 

.. math::

        \dot y_0 = \frac{y_1 - y_0}{h}\quad\longrightarrow\quad y_1 
        = y_0 + h \dot{y}_0\quad.  

Generalizing to any timestep :math:`n`:

.. math::

        y_{n+1} = y_n + h \dot{y}_n \quad.

Whenever all of the :math:`n+1` terms are on one side of the equation and the
:math:`n` terms are on the other, we have an **explicit
numerical method**. This can also be extended to :math:`k` components
for :math:`y_n` with the simple substitution:

.. math::

        \newcommand{b}[1]{\mathbf{#1}}
        y_n \longrightarrow \b X_n
        = \begin{pmatrix}x_1 \\ x_2 \\ \vdots \\ x_k\end{pmatrix},\quad
        \dot{y}_n \longrightarrow \b {\dot X}_n
        = \begin{pmatrix}\dot{x}_1 \\ \dot{x}_2 \\ \vdots \\ \dot{x}_k\end{pmatrix},\quad
        y_{n+1} \longrightarrow \b X_{n+1} = \b X_{n} + h \dot{\b X}_n \quad.

This is intuitively straightforward and easy to implement, but there is a
downside: the solutions **do not converge** for any given timestep. If the
steps are too large, our numerical estimations are essentially dominated by
progation of error and would return results that are non-physical, and if they
are too small the simulation would take too long to run.  

We need a scheme that remains stable and accurate for a wide range of
timesteps, which is what **implicit differentiation** can accomplish. An
example of one such scheme is: 

.. math::

        \b X_{n+1} = \b X_{n} + h \dot{\b X}_{n+1} \quad.

Now we have :math:`n+1` terms on both sides, making this an implicit scheme.
This is know as the `backward Euler method`_ and a common way of solving this
and many other similar schemes that build of off this one is by re-casting it
as a root finding problem. For the backward Euler method, this would look like:

.. _backward Euler method: https://en.wikipedia.org/wiki/Backward_Euler_method

.. math::

        \b g(\b X_{n+1}) = \b X_{n+1} - \b X_n - h \dot{\b X}_{n+1}\quad.

Here, the root of the new function :math:`\b g` is the solution to our original
implicit integration equation. The `Newton-Raphson method
<https://en.wikipedia.org/wiki/Newton%27s_method>`_ is a useful root finding
algorithm, but one of its steps requires the computation of the 
:math:`k \times k` Jacobian: 

.. math::

        \newcommand{\pd}[2]{\frac{\partial#1}{\partial#2}}

        \b{J}(\b {\dot X}_{n+1}) 
        = \pd{\b {\dot X}_{n+1}}{\b X_{n+1}} 
        = \begin{pmatrix} \nabla (\dot x_1)_{n+1} \\ 
                          \nabla (\dot x_2)_{n+1} \\ 
                          \vdots \\
                          \nabla (\dot x_k)_{n+1} \\ 
                         \end{pmatrix} \quad.

.. note:: 
        We have avoided using superscript notation here because that will be
        reserved for identifying iterates in Newton's method, which we discuss
        in later sections. 
        
        ``spacejam`` can also support systems with a different number of
        equations than variables, i.e. non-square Jacobians. See :ref:`diii` .

Accurately computing the elements of the Jacobian can be numerically expensive,
so a method to quickly and accurately compute derivatives would be extremely
useful. ``spacejam`` provides this capability by computing the Jacobian quickly
and accurately via 
`automatic differentiation <Automatic Differentiation: A brief overview_>`__,
which can be used to solve a wide class of problems that depend on implicit
differentiation for numerically stable solutions. 

We walk through using 
``spacejam`` to implement Newton's method for the Backward Euler method and
its slightly more sophisticated sibling, the :math:`s=1` 
`Adams Moulton method`_ in :ref:`examples`. Note: :math:`s=0` is just the
original backward Euler method and :math:`s=1` is also know as the famous
trapezoid rule.

.. _Adams Moulton method: https://en.wikipedia.org/wiki/Linear_multistep_method#Adams%E2%80%93Moulton_methods

.. _ad:

Automatic Differentiation: A brief overview
-------------------------------------------
This is a method to simultaneously compute a function and its derivative to
machine precision. This can be done by introducing the dual number
:math:`\epsilon^2=0`, where :math:`\epsilon\ne0`. If we transform some
arbitrary function :math:`f(x)` to :math:`f(x+\epsilon)` and expand it, we
have: 

.. math::

        f(x+\epsilon) = f(x) + \epsilon f'(x) + O(\epsilon^2)\quad.

By the definition of :math:`\epsilon`, all second order and higher terms in
:math:`\epsilon` vanish and we are left with :math:`f(x+\epsilon) = f(x) +
\epsilon f'(x)`, where the dual part, :math:`f'(x)`, of this transformed
function is the derivative of our original function :math:`f(x)`. If we adhere
to the new system of math introduced by dual numbers, we are able to compute
derivatives of functions exactly. 

For example, multiplying two dual numbers :math:`z_1 = a_r + \epsilon a_d` and 
:math:`z_2 = b_r + \epsilon b_d` would behave like:

.. math::

        z_1 \times z_2 &= (a_r + \epsilon a_d) \times (b_r + \epsilon b_d)
        = a_rb_r + \epsilon(a_rb_d + a_db_r) + \epsilon^2 a_db_d \\
        &= \boxed{a_rb_r + \epsilon(a_rb_d + a_db_r)}\quad.

A function like :math:`f(x) = x^2` could then be automatically differentiated
to give:

.. math::
        f(x) \longrightarrow f(x+\epsilon) 
        = (x + \epsilon) \times (x + \epsilon)
        = x^2 + \epsilon (x\cdot 1 + 1\cdot x) = x^2 + \epsilon\ 2x \quad,

where :math:`f(x) + \epsilon f'(x)` is returned as expected.  Operations like
this can be redefined via **operator overloading**, which we implement in
:ref:`api`. This method is also easily extended to multivariable functions with
the introduction of "dual number basis vectors" 
:math:`\b p_i = i + \epsilon_i 1`, where :math:`i` takes on any of the
components of :math:`\b X_{n}`. For example, the multivariable function
:math:`f(x, y) = xy` would transform like:

.. math::
        \require{cancel}
        x \quad\longrightarrow\quad& \b p_x = x + \epsilon_x + \epsilon_y\ 0 \\
        y \quad\longrightarrow\quad& \b p_y = y + \epsilon_x\ 0 + \epsilon_y \\
        f(x, y) \quad\longrightarrow\quad& f(\b p) = (x + \epsilon_x + \epsilon_y\ 0) 
        \times (y + \epsilon_x\ 0 + \epsilon_y) \\
        &= xy + \epsilon_y x + \epsilon_x y + 
        \cancel{\epsilon_x\epsilon_y} \\
        &= xy + \epsilon_x y + \epsilon_y x \quad,

where we now have:

.. math::
        f(x+\epsilon_x, y+\epsilon_y) 
        &= f(x, y) + \epsilon_x\pd{f}{x} + \epsilon_y\pd{f}{y} 
        = f(x, y) + \epsilon \left[\pd{f}{x},\ \pd{f}{y}\right] \\
        &= f(x, y) + \epsilon \nabla f(x, y)\quad.

This is accomplished internally in ``spacejam.autodiff.Autodiff._ad`` with:

::

        def _ad(self, func, p):                                                                                                     
        """ Internally computes `func(p)` and its derivative(s).                                                                
                                                                                                                                
        Notes                                                                                                                   
        -----                                                                                                                   
        `_ad` returns a nested 1D `numpy.ndarray` to be formatted internally                                                    
        accordingly in :any:`spacejam.autodiff.AutoDiff.__init__`  .                                                            
                                                                                                                                
        Parameters                                                                                                              
        ----------                                                                                                              
        func : numpy.ndarray                                                                                                    
            function(s) specified by user.                                                                                      
        p : numpy.ndarray                                                                                                       
            point(s) specified by user.                                                                                         
        """                                                                                                                     
        if len(p) == 1: # scalar p                                                                                              
            p_mult = np.array([dual.Dual(p)])                                                                                   
                                                                                                                                
        else:# vector p                                                                                                         
            p_mult = [dual.Dual(pi, idx=i, x=p) for i, pi in enumerate(p)]                                                      
            p_mult = np.array(p_mult) # convert list to numpy array                                                             
                                                                                                                                
        result = func(*p_mult) # perform AD with specified function(s)                                                          
        return result

The ``x`` argument in the ``spacejam.dual.Dual`` class above sets the length of
the :math:`\ p` dual basis vector and the ``idx`` argument sets the proper
index to 1 (with the rest being zero).  


