Future
======

``spacejam`` can also be used in a wide range of disciplines, as long as the
system in question that you would like to evolve can be expressed as a series
of differential equations. For example, one use in ecology could be the
modelling the evolution of animal populations, outlined below.

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
Adams-Moulton method gives:

::

        ad_old = sj.AutoDiff(f, X_old)
        D = np.eye(len(ad.r.flatten())) - (h/2)*ad.d
        g = X_iold - X_old - (h/2)*ad.r.flatten() - (h/2)*ad_old.r.flatten()

.. image:: ../figs/test_ii.png

Using the same exact timestep, the phase curve converges for the higher order
scheme.

We would like to see ``spacejam`` applied to a whole host of systems, such as
the one described above. Add more here

