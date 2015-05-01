import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from math import exp,log
from collections import namedtuple

class RateState(object):
    def __init__(self):
        self.mu0 = 0.6
        self.a = 0.005
        self.b = 0.01
        self.dc = 10.
        self.k = 1e-3
        self.v = 1.
        self.vlp = 10.
        self.time = [0]
        self.dispHist = []
        self.results = namedtuple("results",["time","displacement","velocity","friction","state1","state2"])
        self.results.time = []
        self.results.displacement = []
        self.results.velocity = []
        self.results.friction = []
        self.results.state1 = []
        self.results.state2 = []

    def _integrationStep(self, w, t, p):
        self.time.append(t)
        mu, theta, self.v = w
        mu0, vlp, a, b, dc, k = p
        self.v = self.v * exp((mu - mu0 - b * log(self.v * theta / dc)) / a)
        self._vHist.append(self.v)

        dmu_dt = k * (vlp - self.v)
        dtheta_dt = 1. - self.v * theta / dc

        return [dmu_dt,dtheta_dt]

    def solve(self):

        # Parameters from RSF model
        p = [self.mu0,self.vlp,self.a,self.b,self.dc,self.k]

        # Initial conditions at t = 0
        # mu = reference friction value
        # theta = dc/v
        # velocity = v
        w0 = [self.mu0,self.dc/self.v,self.v]

        # Time domain to solve over
        t = np.arange(0,30,0.01)
        self.dispHist = t*self.vlp
        # Integrator settings
        abserr = 1.0e-12
        relerr = 1.0e-12

        # Append initial value to velocity history
        self._vHist = [self.v]

        # Solve it
        wsol = integrate.odeint(self._integrationStep, w0, t, args=(p,),
                                atol=abserr, rtol=relerr)

        #return SimResults(velocity=self._vHist, time=wsol[:,0], friction=wsol[:,1])
        self.results.friction = wsol[:,0]
        self.results.state1 = wsol[:,1]
        self.results.velocity = np.array(self._vHist)

        return self.results
