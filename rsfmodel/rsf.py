import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import integrate
from math import exp, log
from collections import namedtuple

class IncompleteModelError(Exception):
    pass

class StateRelation(object):
    """
    Abstract state relation object that contains the generally used atributes
    in state relations (b,Dc).
    """
    def __init__(self):
        self.b = None
        self.Dc = None
        self.state = None

    def velocity_component(self, system):
        """
        General velocity contribution from a given state variable

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
    def set_steady_state(self, system):
        self.state = self.Dc/system.vref

    def evolve_state(self, system):
        return 1. - system.v * self.state / self.Dc


class RuinaState(StateRelation):
    """
    The slip or Ruina state relation as proposed by Andy Ruina (1983)

    .. math::
    \frac{d\theta}{dt} =  -\frac{V_\text{slider} \theta}{D_c} \text{ln}\left(\frac{V_\text{slider} \theta}{D_c}\right)
    """
    def set_steady_state(self, system):
        self.state = self.Dc/system.vref

    def evolve_state(self, system):
        return -1 * (system.v * self.state / self.Dc) * log(system.v * self.state / self.Dc)


class PrzState(StateRelation):
    """
    The PRZ state relation as proposed by Perrin, Rice, and Zheng (1995):

    .. math::
    \frac{d\theta}{dt} =  1 - \left(\frac{V_\text{slider} \theta}{2D_c}\right) ^2
    """
    def set_steady_state(self, system):
        self.state = 2 * self.Dc / system.v
        self.prz_vref = system.vref/(2*self.Dc)

    def evolve_state(self, system):
        return 1. - (system.v * self.state / (2 * self.Dc))**2

    def velocity_component(self, system):
        """
        Perrin-Rice velocity contribution

        .. math::
        V_\text{contribution} = b \text{ln}\left(V_{\text{prz}0} \theta\right)
        """
        return self.b * np.log(self.prz_vref * self.state)


class NagataState(StateRelation):
    """
    The Nagata state relation as proposed by Nagata et al. (2012):

    .. math::
    \frac{d\theta}{dt} =  1 - \frac{V_\text{slider} \theta}{D_c} - \frac{c}{b}\theta\frac{d\mu}{dt}
    """
    def __init__(self):
        StateRelation.__init__(self)
        self.c = None

    def set_steady_state(self, system):
        self.state = self.Dc / system.vref

    def evolve_state(self, system):
        return 1. - (system.v * self.state / self.Dc) - (self.c / self.b * self.state * system.dmu_dt)


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
        if self.a == None:
            raise IncompleteModelError('Parameter a is None')
        elif self.vref == None:
            raise IncompleteModelError('Parameter vref is None')
        elif self.state_relations == []:
            raise IncompleteModelError('No state relations in state_relations')
        elif self.k == None:
            raise IncompleteModelError('Parameter k is None')
        elif self.time == None:
            raise IncompleteModelError('Parameter time is None')
        elif self.loadpoint_velocity == None:
            raise IncompleteModelError('Parameter loadpoint_velocity is not set')

        for state_relation in self.state_relations:
            if state_relation.b == None:
                raise IncompleteModelError('Parameter b is None')
            elif state_relation.Dc == None:
                raise IncompleteModelError('Parameter Dc is None')

        if len(self.time) != len(self.loadpoint_velocity):
            raise IncompleteModelError('Time and loadpoint_velocity lengths do not match')

    def solve(self, **kwargs):
        """
        Runs the integrator to actually solve the model and returns a
        named tuple of results.
        """
        odeint_kwargs = dict(rtol=1e-12, atol=1e-12)
        odeint_kwargs.update(kwargs)

        # Make sure we have everything set before we try to run
        self.readyCheck()

        # Initial conditions at t = 0
        w0 = [self.mu0]
        for state_variable in self.state_relations:
            state_variable.set_steady_state(self)
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
        self.results.loadpoint_displacement = self._calculateDisplacement(self.loadpoint_velocity)

        # Calculate the slider displacement
        self.results.slider_displacement = self._calculateDisplacement(self.results.slider_velocity)

        return self.results

    def _calculateDisplacement(self, velocity):
        dt = np.ediff1d(self.results.time)
        displacement = np.cumsum(velocity[:-1] * dt)
        displacement = np.insert(displacement, 0, 0)
        return displacement


def phasePlot(system, fig=None, ax1=None):
    """ Make a phase plot of the current model. """
    if fig is None:
        fig = plt.figure(figsize=(8, 7))

    if ax1 is None:
        ax1 = plt.subplot(111)

    v_ratio = np.log(system.results.slider_velocity/system.vref)
    ax1.plot(v_ratio, system.results.friction, color='k', linewidth=2)

    ylims = ax1.get_ylim()
    xlims = ax1.get_xlim()

    # Plot lines of constant a that are in the view
    y_line = system.a * np.array(xlims)
    for mu in np.arange(0, ylims[1], 0.005):
        y_line_plot = y_line + mu
        if max(y_line_plot) > ylims[0]:
            ax1.plot(xlims, y_line_plot, color='k', linestyle='--')

    # Plot a line of rate dependence "Steady State Line"
    state_b_sum = 0
    for state in system.state_relations:
        state_b_sum += state.b
    mu_rate_dependence = system.mu0 + (system.a - state_b_sum)*np.array(xlims)
    ax1.plot(xlims, mu_rate_dependence, color='k', linestyle='--')

    ax1.set_xlabel(r'ln$\frac{V}{V_{ref}}$', fontsize=16, labelpad=20)
    ax1.set_ylabel(r'$\mu$', fontsize=16)
    ax1.set_xlim(xlims)
    ax1.set_ylim(ylims)
    plt.show()
    return fig, ax1

def phasePlot3D(system, fig=None, ax1=None, state_variable=2):
    """ Make a 3D phase plot of the current model. """

    if np.shape(system.results.states)[1] < 2:
        raise ValueError('Must be a multi state-variable system for 3D plotting')

    if fig is None:
        fig = plt.figure(figsize=(8, 7))

    if ax1 is None:
        ax1 = fig.gca(projection='3d')

    v_ratio = np.log(system.results.slider_velocity/system.vref)
    ax1.plot(v_ratio, system.results.states[:,state_variable-1], system.results.friction, color='k', linewidth=2)

    ax1.set_xlabel(r'ln$\frac{V}{V_{ref}}$', fontsize=16)
    ax1.set_ylabel(r'$\theta_%d$' %state_variable, fontsize=16)
    ax1.set_zlabel(r'$\mu$', fontsize=16)
    ax1.xaxis._axinfo['label']['space_factor'] = 2.5
    ax1.yaxis._axinfo['label']['space_factor'] = 2.
    ax1.zaxis._axinfo['label']['space_factor'] = 2.
    plt.show()
    return fig, ax1

def dispPlot(system):
    """ Make a standard plot with displacement as the x variable """
    fig = plt.figure(figsize=(12, 9))
    ax1 = plt.subplot(411)
    ax2 = plt.subplot(412, sharex=ax1)
    ax3 = plt.subplot(413, sharex=ax1)
    ax4 = plt.subplot(414, sharex=ax1)
    ax1.plot(system.results.loadpoint_displacement, system.results.friction, color='k')
    ax2.plot(system.results.loadpoint_displacement, system.results.states, color='k')
    ax3.plot(system.results.loadpoint_displacement, system.results.slider_velocity, color='k')
    ax4.plot(system.results.loadpoint_displacement, system.loadpoint_velocity, color='k')
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
