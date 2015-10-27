import numpy as np
from scipy import integrate
from math import exp
from collections import namedtuple
import warnings


class IncompleteModelError(Exception):
    """
    Special error case for trying to run the model with inadequate information.
    """
    pass


class LoadingSystem(object):
    """ Contains attributes relating to the external loading system """
    def __init__(self):
        self.k = None
        self.time = None  # List of times we want answers at
        self.loadpoint_velocity = None  # Matching list of velocities

    def velocity_evolution(self):
        v_contribution = 0
        for state in self.state_relations:
            v_contribution += state.velocity_component(self)
        self.v = self.vref * exp((self.mu - self.mu0 - v_contribution) / self.a)

    def friction_evolution(self, loadpoint_vel):
        return self.k * (loadpoint_vel - self.v)


class Model(LoadingSystem):
    """ Houses the model coefficients and does the integration """
    def __init__(self):
        LoadingSystem.__init__(self)
        self.mu0 = 0.6
        self.a = None
        self.vref = None
        self.state_relations = []
        self.results = namedtuple("results", ["time", "loadpoint_displacement",
                                              "slider_velocity", "friction",
                                              "states", "slider_displacement"])

    def _integrationStep(self, w, t, system):
        """ Do the calculation for a time-step

        Parameters
        ----------
        w : list
        Current values of integration variables. Friction first, state variables following.
        t : float
        Time at which integration is occurring
        system : model object
        Model that is being solved

        Returns
        -------
        step_results : list
        Results of the integration step. dmu/dt first, followed by dtheta/dt for state variables.
        """

        system.mu = w[0]
        for i, state_variable in enumerate(system.state_relations):
            state_variable.state = w[i+1]

        system.velocity_evolution()

        # Find the loadpoint_velocity corresponding to the most recent time
        # <= the current time.
        loadpoint_vel = system.loadpoint_velocity[system.time <= t][-1]

        self.dmu_dt = system.friction_evolution(loadpoint_vel)
        step_results = [self.dmu_dt]

        for state_variable in system.state_relations:
            dtheta_dt = state_variable.evolve_state(self)
            step_results.append(dtheta_dt)

        return step_results

    def readyCheck(self):
        if self.a is None:
            raise IncompleteModelError('Parameter a is None')
        elif self.vref is None:
            raise IncompleteModelError('Parameter vref is None')
        elif self.state_relations == []:
            raise IncompleteModelError('No state relations in state_relations')
        elif self.k is None:
            raise IncompleteModelError('Parameter k is None')
        elif self.time is None:
            raise IncompleteModelError('Parameter time is None')
        elif self.loadpoint_velocity is None:
            raise IncompleteModelError('Parameter loadpoint_velocity is not set')

        for state_relation in self.state_relations:
            if state_relation.b is None:
                raise IncompleteModelError('Parameter b is None')
            elif state_relation.Dc is None:
                raise IncompleteModelError('Parameter Dc is None')

        if len(self.time) != len(self.loadpoint_velocity):
            raise IncompleteModelError('Time and loadpoint_velocity lengths do not match')

    def _get_critical_times(self, threshold):
        """
        Calculates accelearation and thresholds based on that to find areas
        that are likely problematic to integrate.
        """
        velocity_gradient = np.gradient(self.loadpoint_velocity)
        time_gradient = np.gradient(self.time)
        acceleration = velocity_gradient / time_gradient
        critical_times = self.time[np.abs(acceleration) > threshold]
        return critical_times

    def solve(self, threshold=2, **kwargs):
        """
        Runs the integrator to actually solve the model and returns a
        named tuple of results.

        Parameters
        ----------
        threshold : float
            Threshold used to determine when integration care should be taken. This threshold is
            in terms of maximum load-point acceleration before time step is marked.

        Returns
        -------
        results : named tuple
            Results of the model
        """
        odeint_kwargs = dict(rtol=1e-12, atol=1e-12, mxstep=5000)
        odeint_kwargs.update(kwargs)

        # Make sure we have everything set before we try to run
        self.readyCheck()

        # Initial conditions at t = 0
        w0 = [self.mu0]
        for state_variable in self.state_relations:
            state_variable.set_steady_state(self)
            w0.append(state_variable.state)

        # Find any critial time points we need to let the integrator know about
        self.critical_times = self._get_critical_times(threshold)

        # Solve it
        wsol, self.solver_info = integrate.odeint(self._integrationStep, w0, self.time,
                                                  full_output=True, tcrit=self.critical_times,
                                                  args=(self,), **odeint_kwargs)

        self.results.friction = wsol[:, 0]
        self.results.states = wsol[:, 1:]
        self.results.time = self.time

        # Calculate slider velocity after we have solved everything
        velocity_contribution = 0
        for i, state_variable in enumerate(self.state_relations):
            state_variable.state = wsol[:, i+1]
            velocity_contribution += state_variable.velocity_component(self)

        self.results.slider_velocity = self.vref * np.exp(
                                       (self.results.friction - self.mu0 -
                                        velocity_contribution) / self.a)

        # Calculate displacement from velocity and dt
        self.results.loadpoint_displacement = \
            self._calculateDiscreteDisplacement(self.loadpoint_velocity)

        # Calculate the slider displacement
        self.results.slider_displacement = \
            self._calculateContinuousDisplacement(self.results.slider_velocity)

        # Check slider displacement for accumulated error and warn
        if not self._check_slider_displacement():
            warnings.warn("Slider displacement differs from prediction by over "
                          "1%. Smaller requested time resolution should be used "
                          "If you intend to use the slider displacment output.")

        return self.results

    def _check_slider_displacement(self, tol=0.01):
        """
        Checks that the slider displacement total is within a given tolerance
        of the prediction from steady-state theory. Defaults to 1%
        """
        a_minus_b = self.a
        for state_relation in self.state_relations:
            a_minus_b -= state_relation.b

        dmu = a_minus_b * np.log(self.results.slider_velocity[-1]/self.vref)
        dx = -dmu/self.k

        predicted_slider_displacement = self.results.loadpoint_displacement[-1] + dx
        actual_slider_diaplacement = self.results.slider_displacement[-1]

        difference = np.abs(predicted_slider_displacement - actual_slider_diaplacement)/\
                     predicted_slider_displacement

        if difference > tol:
            return False
        else:
            return True

    def _calculateContinuousDisplacement(self, velocity):
        return integrate.cumtrapz(velocity, self.time, initial=0)

    def _calculateDiscreteDisplacement(self, velocity):
        dt = np.ediff1d(self.results.time)
        displacement = np.cumsum(velocity[:-1] * dt)
        displacement = np.insert(displacement, 0, 0)
        return displacement
