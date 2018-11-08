Background
==========

Euler's Method As a Solution to Our Problem
-------------------------------------------
Euler's method is a numerical procedure for solving ordinary differential
equations with a given initial value. In the case of our problem statement,
here is how it applies:

* We have some initial conditions (such as position and velocity of a planet)
  and we want to know what kind of orbit this planet will trace out, given that
  only the force acting on it is gravity. 
  
* Using the physical insight that
  the "slope" of position over time is velocity, and the "slope" of velocity
  over time is acceleration, we can predict, or integrate, how the quantities
  will change over time.  

* More explicitly, we can use the acceleration
  supplied by gravity to predict the velocity of our planet, and then use this
  velocity to predict its position a timestep :math:`Delta t` later.  
  
* This gives us a new position and the whole process starts over again at the
  next timestep. Here is a schematic of the Euler integration method.

.. image:: http://jmahaffy.sdsu.edu/courses/f00/math122/lectures/num_method_diff_equations/images/euler_ani.gif

This plot above could represent the component of the planet's velocity varies
over time. Specifically, we have some solution curve (black) that we want to
approximate (red), given that we only know two things:

* where we started :math:`(t_0, y_0)`

* the rate of how where we were changes with time 
  :math:`\left(y'_0 = \frac{y_1 - y_0}{\Delta t}\right)`

The cool thing about this is that even though we do not explicity know what
:math:`y_1` is, the fact that we are given :math:`y'_0` from the initial
conditions allows us to bootstrap our way around this. Starting with the
definition of slope, we can use the timestep :math:`h = \Delta t`, to find
where we will be a timestep later :math:`y_1`: 

.. math::

        y_0' = \frac{y_1 - y_0}{\Delta t}\quad\longrightarrow\quad y_1 
        = y_0 + \Delta t\ y_0'\quad.  

Generalizing to any timestep :math:`n`:

.. math::

        y_{n+1} = y_n + \Delta t\ y_n'

Whenever all of the :math:`n+1` terms are on one side of the equation and the
:math:`n` terms are on the other, we have something called an **explicit
numerical method**. This is intuitively straightforward and easy to implement,
but there is a downside, the solutions **do not converge** for a given
timestep. A solution to this would be to consider an implicit analogue to
Euler's method. 

Instead, we can recast our problem in the form of an oscillator with some
motion :math:`x(t)` along the x-axis confined to an area that is close to this
axis. In other words,

.. math::

        \newcommand{b}[1]{\mathbf#1}
        \newcommand{od}[2]{\frac{\mathrm{d}#1}{\mathrm{d}#2}}
        \b f(\b X_n) = \dot{\b X}_n = \od{}{t}\begin{bmatrix}x_n\\y_n\end{bmatrix}
        = \begin{bmatrix}-x_n\\-ky_n\end{bmatrix}
        \quad,

for large :math:`k` to constrain our oscillator to the x-axis.

Running our explicit Euler scheme then gives,

.. math::

        \b X_{n+1} = \b X_n + h\b f(\b X_n) 
        = \begin{pmatrix}x_0\\y_0\end{pmatrix}
        - h\begin{pmatrix}-x_0\\-ky_0\end{pmatrix}
        = \begin{bmatrix}(1 - h)x_0\\(1 - hk) y_0\end{bmatrix}
        \quad.

**This is our problem**. If :math:`|1 - hk| > 0`, then :math:`|y_{n+1}| >
|y_n|`, which would imply that our spring would actually get farther and
farther away from the x-axis! In other words, if we happened to select a
timestep larger than :math:`h = 2/k`, our solution would not converge. By
definition, :math:`k` is large, so the only way to avoid having this system
blow up on us is to select very small timesteps, which is computationally
expensive and would take forever to run. 

We need a scheme that will remain stable for a wide range of timesteps, which
is what **implicit differentiation can accomplish**. One way of approaching this
is to re-cast it as the root finding problem

.. math::

        \b g(\b X_{n+1}) = \b X_{n+1} - \b X_n - h\b f(\b X_{n+1})\quad.

Here, the root of the new function :math:`\b g` is the solution to our original
implicit integration problem. The `Newton-Raphson method
<https://en.wikipedia.org/wiki/Newton%27s_method>`_. is a useful root finding
algorithm, but it requires the computation of the Jacobian,

.. math::

        \b I - h\b J\b f(\b X_{n+1})\quad.

Accurately computing the elements of the Jacobian can be numerically expensive,
so a method to quickly and accurately compute derivatives would be extremely
useful.

``spacejam`` provides this capability by computing the Jacobian quickly and
accurately via 
`automatic differentiation <Automatic Differentiation: A brief overview_>`__,
which can be used to solve a class or problems that depend on implicit
differentiation for numerically stable solutions.

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
to the new system of algebra introduced by dual numbers, we are able to compute
derivatives of functions exactly. 

For example, multiplying two dual numbers :math:`z_1 = a_r + \epsilon a_d` and 
:math:`z_2 = b_r + \epsilon b_d` would behave like:

.. math::

        z_1 \times z_2 &= (a_r + \epsilon a_d) \times (b_r + \epsilon b_d)
        = a_rb_r + \epsilon(a_rb_d + a_db_r) + \epsilon^2 a_db_d \\
        &= \boxed{a_rb_r + \epsilon(a_rb_d + a_db_r)}\quad.

Operations like this can be redefined via **operator overloading**, which we
implement in :ref:`api`.

