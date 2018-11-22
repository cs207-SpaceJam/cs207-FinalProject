.. _examples:

Example Applications
====================
``spacejam`` can be used to simulate a wide range of physical systems. Below
are two applications of what you can do with ``spacejam`` and its automatic
differentiation capabilities. These applications use the 
`backward Euler`_ and :math:`(s=1)` `Adam-Moulton`_ method.

.. _backward Euler: https://en.wikipedia.org/wiki/Backward_Euler_method
.. _`Adam-Moulton`: https://en.wikipedia.org/wiki/Linear_multistep_method#Adams%E2%80%93Moulton_methods

Backward Euler and Newton-Raphson
---------------------------------
As we saw in :ref:`numerical`, this is a numerical scheme to solve the implicit
equation:

.. math::
        \newcommand{b}[1]{\mathbf{#1}}

        \b X_{n+1} = \b X_{n} + h \dot{\b X}_{n+1}

by re-casting it as the root finding problem:

.. math::
        \b g(\b X_{n+1}) = \b X_{n+1} - \b X_n - h \dot{\b X}_{n+1}\quad.


In 1D, the Newton-Raphson method successively finds better and better
approximations to the root of a function :math:`f(x)` in the following way:

.. image:: https://upload.wikimedia.org/wikipedia/commons/e/e0/NewtonIteration_Ani.gif

- Make a guess next to one of the roots you want
- Draw the corresponding tangent line (red) at this guess evaluated on the
  original function (blue) 
- Trace this tangent line to where it intercepts the x-axis
- Make this intercept your new guess
- Rinse and repeat until your latest guess is close enough (your tolerance) to
  the root you wanted to approximate in the first place

The equation for this can be quickly derived by solving for the next root
iterate :math:`x_{n+1}` from the definition of the derivative:

.. math::
        \text{slope} = \frac{\text{rise}}{\text{run}}\quad\longrightarrow\quad 
        f'(x_n) = \frac{f(x_n)}{x_n - x_{n+1}}\quad\longrightarrow\quad
        x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)} \quad .

This is naturally extended to vector functions that accept multi-valued input by
using the multi-variable version of the derivative, the Jacobian :math:`\b J`:

.. math::
        \newcommand{\pd}[2]{\frac{\partial#1}{\partial#2}}

        \b X_{n+1} = \b X_{n} - \b J[\b f(\b X_n)]^{-1} \b f(\b X_n) \quad,

where:

.. math::
        \b J[\b f(\b X_n)]_{ij} &= \pd{f_i}{x_j} \quad, \\
        \b f\left(\b X_n\right) &= [f_1, f_2, \cdots, f_m] \quad, \\
        \b X_n &= [x_1, x_2, \cdots, x_k] \quad.

For these examples, :math:`m=k`, 
:math:`\b f = \b {\dot X}_n = [\dot x_1, \dot x_2, \cdots, \dot x_m]_n`,
:math:`1 \le i,j \le k` .

Applying this to our backward Euler equation:

.. math::
        0 &= \b g\left(\b X_{n+1}\right) = \b X_{n+1} - \b X_n - h \dot{\b X}_{n+1} \quad, \\
        \b X_{n+1}^{(i+1)} &= \b X_{n+1}^{(i)} - 
        \b D\left[\b g\left(\b X_{n+1}\right)^{(i)}\right]^{-1} \b g\left(\b X_{n+1}\right)^{(i)} \quad.

Here, :math:`(i)` and :math:`(i+1)` have been used to avoid confusion with the
:math:`n` and :math:`n+1` iterate used in the 1D example above, and the root to
this equation is the solution :math:`\b X_{n+1}` to our original implicit equation.
The Jacobian :math:`\b J` is hiding inside of :math:`\b D` and we can make it show itself by
just performing the multi-variable derivative that is required of the
Newton-Raphson method:

.. math::
        \require{cancel}
        \b D\left[\b g\left(\b X_{n+1}\right)^{(i)}\right] 
        = \pd{\b g\left(\b X_{n+1}\right)^{(i)}}{\b X_{n+1}^{(i)}}
        = \pd{\b X_{n+1}^{(i)}}{\b X_{n+1}^{(i)}} 
        - \cancelto{0}{\pd{\b X_{n}^{(i)}}{\b X_{n+1}^{(i)}}}
          - \pd{h \b {\dot X}_{n+1}}{\b X_{n+1}^{(i)}}
        = \b I - h\b{J}\left[\left(\b {\dot X}_{n+1}\right)^{(i)}\right] \quad.


All that is needed now is an initial guess for :math:`\b X_{n+1}^{(0)}` to jump
start Newton's method. A single forward Euler step should do:

.. math::
        \b X_{n+1}^{(0)} &= \b X_{n}^{(0)} + h \b {\dot X}_n^{(0)}\quad, \\
        \b {\dot X}_n^{(0)}
        &= \begin{bmatrix}
                \dot{x}_1 (t=0) \\
                \dot{x}_2 (t=0) \\
                \vdots \\
                \dot{x}_k (t=0)
        \end{bmatrix}\quad. 

In this framework, both the real and dual part of the dual object returned by
``spacejam`` will be used. To summarize:

- The user supplies the system of equations :math:`\b {\dot X}_{n}`
  and initial conditions :math:`\b {\dot X}_{n}^{(0)}` .

- The user implements the integration scheme using the real part returned
  from ``spacejam`` for :math:`\b {\dot X}_{n}^{(i)}` and the dual part
  as :math:`\b{J}\left[\left(\b {\dot X}_{n+1}\right)^{(i)}\right]` . 
  
:math:`(s=1)` Adam-Moulton and Newton-Raphson
---------------------------------------------
A similar implementation can be made with the next order up in this family of
implicit methods. In this scheme we have:

.. math::
        \b X_{n+1} = \b X_n + \frac{1}{2}h\left(\b {\dot X_{n+1}} + \b X_n\right)\quad.

Applying the same treatment of turning this into a root finding problem and
applying Newton's method gives the similar result:

.. math::
        \b g(\b X_{n+1}) &= \b X_{n+1} - \b X_n 
        - \frac{h}{2} \b {\dot X_{n+1}} 
        - \frac{h}{2} \b {\dot X_n} \quad, \\
        \b X_{n+1}^{(i+1)} &= \b X_{n+1}^{(i)} 
        - \b D\left[\b g\left(\b X_{n+1}\right)^{(i)}\right]^{-1}
          \b g\left(\b X_{n+1}\right)^{(i)} \quad, \\
        \b D &= \left[ \b I 
        - \frac{h}{2}\b J\left(\b {\dot X_{n+1}}\right)^{(i)}\right] \quad .

In this new scheme, :math:`\b D` has an extra factor of :math:`1/2` on its
Jacobian in the backward and now ``spacejam`` will also be computing 
:math:`\b {\dot X_n}` . 



Lotka–Volterra Equations
------------------------
These are a popular system of differential equations used to describe the
`dynamics of biological systems`_ where two sets of species (predator and prey)
interact. The population of each can then be tracked with:

.. _dynamics of biological systems: https://en.wikipedia.org/wiki/Lotka%E2%80%93Volterra_equations

.. math::
        \newcommand{\od}[2]{\frac{\mathrm d #1}{\mathrm d #2}}
        \newcommand{\pd}[2]{\frac{\partial#1}{\partial#2}}

        \od{x}{t} &= \dot x = \alpha x - \beta xy\quad, \\
        \od{y}{t} &= \dot y = \delta xy - \gamma y\quad,

where,

- :math:`x`: number of prey
- :math:`y`: number of predators
- :math:`\dot x` and :math:`\dot y`: instantaneous growth rate of the prey and
  predator populations, respectively 
- TODO: say more about :math:`\alpha,\beta,\delta,\gamma`

Simulating how these two populations evolve over time with backward Euler
could then be accomplished with:

::

        import spacejam as sj

        # some useful packages for visualization
        import matplotlib.pyplot as plt
        import numpy as np
        import seaborn as sns

        sns.set_style('darkgrid')

        def f(x1, x2, alpha=4., beta=4., delta=1., gamma=1.):
                f1 = alpha*x1 - beta*x1*x2
                f2 = delta*x1*x2 - gamma*x2
                return np.array([f1, f2])

        # run simulation
        X_old = np.array([2., 1.]) # initial population conditions ([prey, predator])
        h = .01 # timestep
        t = 0 # number of steps
        T = 10

        time, x1_sols, x2_sols = [], [], [] # hold solutions for plotting later
        while t < T:
            time.append(t)
            x1_sols.append(X_old[0])
            x2_sols.append(X_old[1])

            # use forward Euler for initial guess for X_n+1 = X_new
            X_new = X_old + h*f(*X_old) # X_{n+1}^(0)

            # Iterate to better solution for X_new using Newton-Raphson method
            # on backward Euler implementation
            X_iold = X_old
            X_inew = X_new
            idx, idx_break = 0, 1E4
            while np.linalg.norm(X_inew - X_iold) > 1E-14:
                X_iold = X_inew
        
                if idx > idx_break:
                        sys.exit('solution did not converge')
            
                ad = sj.AutoDiff(f, X_iold) # get Jacobian (ad.d)
                D = np.eye(len(ad.r.flatten())) - h*ad.d
                g = X_iold - X_old - h*ad.r.flatten()
                X_inew = X_iold - np.dot(np.linalg.pinv(D), g)
                # update
                idx += 1

            # update
            X_new = X_iold
            X_old = X_new
            t += h

And visualized with:

::

            # plot setup
            fig, axes = plt.subplots(1, 2, figsize=(10, 3))
            ax1, ax2 = axes

            # solution plot
            ax1.plot(time, x1_sols, label='prey')
            ax1.plot(time, x2_sols, label='predator')
            ax1.set_xlabel('time (arbitrary units)')
            ax1.set_ylabel('population')
            ax1.legend(ncol=2)

            # phase plot
            ax2.plot(x1_sols, x2_sols)
            ax2.set_xlabel('prey population')
            ax2.set_ylabel('predator population')

            plt.suptitle('Lotka Volterra System Example')

.. image:: ../figs/test.png




In comparison, making the necessary modifications for the :math:`s=1`
Adam-Moulton method gives:

::

        ad_old = sj.AutoDiff(f, X_old)
        D = np.eye(len(ad.r.flatten())) - (h/2)*ad.d
        g = X_iold - X_old - (h/2)*ad.r.flatten() - (h/2)*ad_old.r.flatten()

.. image:: ../figs/test_ii.png

Using the same exact timestep, the phase curve converges for the higher order
scheme.




