import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from math import exp, log
from collections import namedtuple


class StateRelation(object):
    """
    Abstract state relation object that contains the generally used atributes
    in state relations and the contribution to slider velocity for each state.
    """
    def __init__(self, relation):
        self.b = None
        self.Dc = None
        self.state = None

    def velocity_component(self, system):
        """
        General velocity contribution from this state variable

        .. math::
        V_\text{contribution} = b \text{ln}\left(\frac{V_0 \theta}{D_c}\right)
        """
        return self.b * np.log(system.vref * self.state / self.Dc)


class DieterichState(StateRelation):
    """
    The slowness or Dieterich state relation as proposed by Jim Dieterich (1979)


    .. math::
    \frac{d\theta}{dt} = 1 - \frac{V_\text{slider} \theta}{D_c}
    """
    def _set_steady_state(self, system):
        self.state = self.Dc/system.vref

    def evolve_state(self, system):
        if self.state is None:
            self.state = _steady_state(self, system)

        return 1. - system.v * self.state / self.Dc


class RuinaState(StateRelation):
    """
    The slip or Ruina state relation as proposed by Andy Ruina (1983)

    .. math::
    \frac{d\theta}{dt} =  -\frac{V_\text{slider} \theta}{D_c} \text{ln}\left(\frac{V_\text{slider} \theta}{D_c}\right)
    """
    def _set_steady_state(self, system):
        self.state = self.Dc/system.vref

    def evolve_state(self, system):
        if self.state is None:
            self.state = _steady_state(self, system)

        return -1 * (system.v * self.state / self.Dc) * log(system.v * self.state / self.Dc)


class PrzState(StateRelation):
    """
    The PRZ state relation as proposed by Perrin, Rice, and Zheng (1995):

    .. math::
    \frac{d\theta}{dt} =  1 - \left(\frac{V_\text{slider} \theta}{2D_c}\right) ^2
    """
    def _set_steady_state(self, system):
        self.state = 2 * self.Dc / system.vref

    def evolve_state(self, system):
        if self.state is None:
            self.state = _steady_state(self, system)
        # return dtheta/dt
        return 1. - (system.v * self.state / (2 * self.Dc))**2


class NagataState(StateRelation):
    """
    The Nagata state relation as proposed by Nagata et al. (2012):

    .. math::
    \frac{d\theta}{dt} =  1 - \frac{V_\text{slider} \theta}{D_c} - \frac{c}{b}\theta\frac{d\mu}{dt}
    """
    def __init__(self):
        self.c = None

    def _set_steady_state(self, system):
        self.state = self.Dc / system.vref

    def evolve_state(self, system):
        if self.state is None:
            self.state = _steady_state(self, system)
        # return dtheta/dt
        return 1. - (system.v * self.state / self.Dc) - (self.c / self.b * self.state * system.dmu_dt)

class LoadingSystem(object):
    """ Contains attributes relating to the external loading system """
    def __init__(self):
        self.k = None
        self.time = None  # List of times we want answers at
        self.loadpoint_velocity = []  # Matching list of velocities

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
        self.mu0 = None
        self.a = None
        self.vref = None
        self.slider_velocity = None
        self.state_relations = []
        self.results = namedtuple("results", ["time", "displacement",
                                              "slider_velocity", "friction",
                                              "states"])

    def _integrationStep(self, w, t, system):
        """ Do the calculation for a time-step """

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
        return True

    def solve(self, **kwargs):
        """
        Runs the integrator to actually solve the model and returns a
        named tuple of results.
        """
        odeint_kwargs = dict(rtol=1e-12, atol=1e-12)
        odeint_kwargs.update(kwargs)

        # Make sure we have everything set before we try to run
        if self.readyCheck() != True:
            raise RuntimeError('Not all model parameters set')

        # Initial conditions at t = 0
        w0 = [self.mu0]
        for state_variable in self.state_relations:
            state_variable._set_steady_state(self)
            w0.append(state_variable.state)

        # Solve it
        wsol = integrate.odeint(self._integrationStep, w0, self.time,
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
        dt = np.ediff1d(self.time)
        self.results.displacement = np.cumsum(self.loadpoint_velocity[:-1] * dt)
        self.results.displacement = np.insert(self.results.displacement, 0, 0)

        return self.results


def phasePlot(system):
    """ Make a phase plot of the current model. """
    fig = plt.figure()
    ax1 = plt.subplot(111)
    v_ratio = np.log(system.results.slider_velocity/system.vref)
    ax1.plot(v_ratio, system.results.friction, color='k')
    ax1.set_xlabel('Log(V/Vref)')
    ax1.set_ylabel('Friction')
    plt.show()


def dispPlot(system):
    """ Make a standard plot with displacement as the x variable """
    fig = plt.figure(figsize=(12, 9))
    ax1 = plt.subplot(411)
    ax2 = plt.subplot(412, sharex=ax1)
    ax3 = plt.subplot(413, sharex=ax1)
    ax4 = plt.subplot(414, sharex=ax1)
    ax1.plot(system.results.displacement, system.results.friction, color='k')
    ax2.plot(system.results.displacement, system.results.states, color='k')
    ax3.plot(system.results.displacement, system.results.slider_velocity, color='k')
    ax4.plot(system.results.displacement, system.loadpoint_velocity, color='k')
    ax1.set_ylabel('Friction')
    ax2.set_ylabel('State')
    ax3.set_ylabel('Slider Velocity')
    ax4.set_ylabel('Loadpoint Velocity')
    ax4.set_xlabel('Displacement')
    plt.show()


def timePlot(system):
    """ Make a standard plot with time as the x variable """
    fig = plt.figure(figsize=(12, 9))
    ax1 = plt.subplot(411)
    ax2 = plt.subplot(412, sharex=ax1)
    ax3 = plt.subplot(413, sharex=ax1)
    ax4 = plt.subplot(414, sharex=ax1)
    ax1.plot(system.results.time, system.results.friction, color='k')
    ax2.plot(system.results.time, system.results.states, color='k')
    ax3.plot(system.results.time, system.results.slider_velocity, color='k')
    ax4.plot(system.results.time, system.loadpoint_velocity, color='k')
    ax1.set_ylabel('Friction')
    ax2.set_ylabel('State')
    ax3.set_ylabel('Slider Velocity')
    ax4.set_ylabel('Loadpoint Velocity')
    ax4.set_xlabel('Time')
    plt.show()
