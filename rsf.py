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
from math import exp,log
from collections import namedtuple

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
        self.vlp = None
        self.vref = None
        self.model_time = None # List of times we want answers at
        # Results of running the model
        self.results = namedtuple("results",["time","displacement",
                                             "slider_velocity","friction",
                                             "state1"])
        # Integrator settings
        self.abserr = 1.0e-12
        self.relerr = 1.0e-12
        self.loadpoint_velocity = []

    def _integrationStep(self, w, t):
        """
        Do the calculation for a time-step
        """
        self.mu, self.theta, self.v = w

        # Not sure that this is the best way to handle this, but it's a start
        # Take the time and find the time in our model_times that is the
        # last one smaller than it
        i = np.argmax(self.model_time>t) - 1

        self.v = self.vref * exp((self.mu - self.mu0 - self.b *
                                  log(self.vref * self.theta / self.dc)) / self.a)

        dmu_dt = self.k * (self.loadpoint_velocity[i] - self.v)
        dtheta_dt = 1. - self.v * self.theta / self.dc

        return [dmu_dt,dtheta_dt]

    def readyCheck(self):
        return True

    def solve(self):
        """
        Runs the integrator to actually solve the model and returns a
        named tuple of results.
        """
        # Make sure we have everything set before we try to run
        if self.readyCheck() != True:
            raise RuntimeError('Not all model parameters set')

        # Initial conditions at t = 0
        # mu = reference friction value, theta = dc/v, velocity = v
        self.theta = self.dc/self.v
        w0 = [self.mu0,self.theta,self.v]

        # Solve it
        wsol = integrate.odeint(self._integrationStep, w0, self.model_time,
                                atol=self.abserr, rtol=self.relerr)

        self.results.friction = wsol[:,0]
        self.results.state1 = wsol[:,1]
        self.results.slider_velocity = self.vref * np.exp((self.results.friction - self.mu0 - self.b * np.log(self.vref * self.results.state1 / self.dc)) / self.a)
        self.results.time = self.model_time

        # Calculate displacement from velocity and dt
        dt = np.ediff1d(self.model_time)
        self.results.displacement = np.cumsum(self.loadpoint_velocity[:-1] * dt)
        self.results.displacement = np.insert(self.results.displacement,0,0)

        return self.results

    def phasePlot(self):
        """
        Make a phase plot of the current model.
        """
        # Need to make sure the model has run! Duh!

        fig = plt.figure()
        ax1 = plt.subplot(111)
        ax1.plot(np.log(self.results.slider_velocity/self.vref),self.results.friction,color='k')
        ax1.set_xlabel('Log(V/Vref)')
        ax1.set_ylabel('Friction')
        plt.show()
