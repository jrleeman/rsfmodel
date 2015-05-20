#!/usr/bin/env python

"""
rsf

This module provides rate and state frictional modeling capability.

Documentation is provided throughout the module in the form of docstrings.
It is easy to view these in the iPython interactive shell environment. Simply
type the command and ? to view the docstring. Examples are provided at the
GitHub page (https://github.com/jrleeman/rate-and-state) in the README.md file.
"""

__authors__ = ["John Leeman", "Ryan May"]
__credits__ = ["Chris Marone", "Demian Saffer"]
__license__ = ""
__version__ = "1.0."
__maintainer__ = "John Leeman"
__email__ = "kd5wxb@gmail.com"
__status__ = "Development"

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from math import exp, log
from collections import namedtuple


def dieterichState(model):
    return 1. - model.v * model.theta / model.dc


def ruinaState(model):
    return -1*(model.v * model.theta / model.dc)*log(model.v * model.theta / model.dc)


def przState(model):
    return 1. - (model.v * model.theta / (2*model.dc))**2


class RateState(object):
    """
    Create a model for frictional behavior
    """
    def __init__(self):
        # Rate and state model parameters
        self.mu0 = None
        self.a = None
        self.b = None
        self.dc = None
        self.k = None
        self.v = None
        # self.vlp = None
        self.vref = None
        self.model_time = None  # List of times we want answers at
        # Results of running the model
        self.results = namedtuple("results", ["time", "displacement", "slider_velocity", "friction", "state1"])
        # Integrator settings
        self.loadpoint_velocity = []
        self.stateLaw = None

    def _integrationStep(self, w, t):
        """
        Do the calculation for a time-step
        """
        self.mu, self.theta = w

        self.v = self.vref * exp((self.mu - self.mu0 - self.b *
                                  log(self.vref * self.theta / self.dc)) / self.a)

        # Find the loadpoint_velocity corresponding to the most recent time
        # <= the current time.
        loadpoint_vel = self.loadpoint_velocity[self.model_time <= t][-1]
        dmu_dt = self.k * (loadpoint_vel - self.v)
        dtheta_dt = self.stateLaw(self)

        return [dmu_dt, dtheta_dt]

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
        # mu = reference friction value, theta = dc/v, velocity = v
        self.theta = self.dc/self.v
        w0 = [self.mu0, self.theta]

        # Solve it
        wsol = integrate.odeint(self._integrationStep, w0, self.model_time, **odeint_kwargs)

        self.results.friction = wsol[:, 0]
        self.results.state1 = wsol[:, 1]
        self.results.slider_velocity = self.vref * np.exp((self.results.friction - self.mu0 - self.b * np.log(self.vref * self.results.state1 / self.dc)) / self.a)
        self.results.time = self.model_time

        # Calculate displacement from velocity and dt
        dt = np.ediff1d(self.model_time)
        self.results.displacement = np.cumsum(self.loadpoint_velocity[:-1] * dt)
        self.results.displacement = np.insert(self.results.displacement, 0, 0)

        return self.results

    def phasePlot(self):
        """
        Make a phase plot of the current model.
        """
        # Need to make sure the model has run! Duh!

        fig = plt.figure()
        ax1 = plt.subplot(111)
        ax1.plot(np.log(self.results.slider_velocity/self.vref), self.results.friction, color='k')
        ax1.set_xlabel('Log(V/Vref)')
        ax1.set_ylabel('Friction')
        plt.show()

    def dispPlot(self):
        """
        Make a standard plot with displacement as the x variable
        """
        fig = plt.figure(figsize=(12, 9))
        ax1 = plt.subplot(411)
        ax2 = plt.subplot(412, sharex=ax1)
        ax3 = plt.subplot(413, sharex=ax1)
        ax4 = plt.subplot(414, sharex=ax1)
        ax1.plot(self.results.displacement, self.results.friction, color='k')
        ax2.plot(self.results.displacement, self.results.state1, color='k')
        ax3.plot(self.results.displacement, self.results.slider_velocity, color='k')
        ax4.plot(self.results.displacement, self.loadpoint_velocity, color='k')
        ax1.set_ylabel('Friction')
        ax2.set_ylabel('State')
        ax3.set_ylabel('Slider Velocity')
        ax4.set_ylabel('Loadpoint Velocity')
        ax4.set_xlabel('Displacement')
        plt.show()

    def timePlot(self):
        """
        Make a standard plot with time as the x variable
        """
        fig = plt.figure(figsize=(12, 9))
        ax1 = plt.subplot(411)
        ax2 = plt.subplot(412, sharex=ax1)
        ax3 = plt.subplot(413, sharex=ax1)
        ax4 = plt.subplot(414, sharex=ax1)
        ax1.plot(self.results.time, self.results.friction, color='k')
        ax2.plot(self.results.time, self.results.state1, color='k')
        ax3.plot(self.results.time, self.results.slider_velocity, color='k')
        ax4.plot(self.results.time, self.loadpoint_velocity, color='k')
        ax1.set_ylabel('Friction')
        ax2.set_ylabel('State')
        ax3.set_ylabel('Slider Velocity')
        ax4.set_ylabel('Loadpoint Velocity')
        ax4.set_xlabel('Time')
        plt.show()
