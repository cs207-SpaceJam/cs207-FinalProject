Future
======

TODO: merge with application sections

``spacejam`` has been shown to be capable of solving systems of equations in
:ref:`diii`. For our future work, we will apply this functionality to integrate
2D orbits in a simple two body planetary system. As previously
mentioned, we need an integration scheme that is insensitive to the size of the timestep and also has a stable solution. The Adams–Moulton method is one such approach that can accomplish this because it is an implicit linear multistep scheme. This method also has the added benefit of not needing additional equations to be solved as the number of timesteps increases [Süli & Mayers 2003, p. 353]. 

An exciting challenge will be extending the simple application to a 1D
spring, shown below, to a coupled 2D oscillating system that characterizes orbital motion. A possible way to address this would be to store the Cartesian components of the planet's motion position, velocity, and acceleration as solution vectors, and just iterate each component. 

::

        import matplotlib.pyplot as plt
        import numpy as np
        import seaborn as sns

        sns.set_style('darkgrid')
        %config InlineBackend.figure_format = 'retina'

        # initial conditions
        x_old = 10
        v_old = 0
        k = 1
        m = 1
        t_max = 100
        h = .1 # timestep

        xs = [x_old] # holds positions
        vs = [v_old] # holds veclocities

        # run simulation
        t = 0
        while t < t_max:
            # implicit Euler method
            delta_v = -h*k*(x_old + h) / (m * (1 + h**2))
            v_new = v_old + delta_v
            delta_x = h*v_new
            x_new = x_old + delta_x
            
            # Euler-Cromer (semi-implicit method)
            #v_new = v_old - h*(k/m)*x_old
            #x_new = x_old + h*v_new
            
            # update
            xs.append(x_new)
            vs.append(v_new)
            x_old = x_new
            v_old = v_new
            t += h

        >>> plt.plot(xs)
        >>> plt.xlabel('step')
        >>> plt.ylabel('x')
        >>> plt.title('Backward Euler Method')

.. image:: ../_images/spring.png
   :align: left

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
  predator populations, respectively .
- :math:`\gamma`: predator mortality rate
- :math:`\alpha`: intrinsic rate of prey population increase
- :math:`\beta`: predation rate coefficient
- :math:`\delta`:reproduction rate of predators per 1 prey eaten

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


