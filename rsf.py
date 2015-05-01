import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from math import exp,log

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
        return [self._vHist, wsol[:,0], wsol[:,1]]

model = RateState()
velocity,friction,state = model.solve()

print np.shape(np.arange(0,30,0.01))
print np.shape(velocity)
print np.shape(friction)
print np.shape(state)

t = np.arange(0,30,0.01)

# Read in Chris Marone's Solutions
cjm_disp, cjm_mu = np.loadtxt('cjm_model.dis',skiprows=2,unpack=True)
cjm_time, cjm_mu = np.loadtxt('cjm_model.tim',skiprows=2,unpack=True)
cjm_state = np.loadtxt('cjm_model.sta',skiprows=2,unpack=True,usecols=[1])
cjm_vel = np.loadtxt('cjm_model.vel',skiprows=2,unpack=True,usecols=[0])
cjm_vel = np.exp(cjm_vel)*1.


# velocities = []
# vel = 1.
# new_vel = 1.
# for fric,tht in zip(friction,state):
#     new_vel = vel*np.exp((fric-0.6-0.01*np.log(vel*tht/10.))/0.005)
#     velocities.append(new_vel)
#     vel = new_vel

fig = plt.figure(1)
plt.title('Slider Velocity')
plt.plot(model.time,velocity,color='k',label='JRL')
plt.plot(cjm_time[1:],cjm_vel,color='r',label='CJM')
#plt.plot(velocities,color='b')

fig = plt.figure(2)
plt.title('Friction')
plt.plot(t,friction,color='k',label='JRL')
plt.plot(cjm_time,cjm_mu,color='r',label='CJM')

fig = plt.figure(3)
plt.title('State Variable')
plt.plot(t,state,color='k',label='JRL')
plt.plot(cjm_time,cjm_state,color='r',label='CJM')

fig = plt.figure(4)
plt.title('Friction')
plt.plot(model.dispHist,friction,color='k',label='JRL')
plt.plot(cjm_disp,cjm_mu,color='r',label='CJM')

plt.show()
