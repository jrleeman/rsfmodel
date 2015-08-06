import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
    ax1.plot(v_ratio, system.results.states[:, state_variable-1],
             system.results.friction, color='k', linewidth=2)

    ax1.set_xlabel(r'ln$\frac{V}{V_{ref}}$', fontsize=16)
    ax1.set_ylabel(r'$\theta_%d$' % state_variable, fontsize=16)
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
