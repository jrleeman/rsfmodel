import rsf
import numpy as np
import matplotlib.pyplot as plt

model = rsf.RateState()

# Set model initial conditions
model.mu0 = 0.6
model.a = 0.005
model.b = 0.01
model.dc = 10.
model.k = 1e-3
model.v = 1.
model.vlp = 10.

# We want to solve for 30 seconds at 100Hz
model.model_time = np.arange(0,30.01,0.01)


results = model.solve()

velocity = results.velocity
friction = results.friction
state = results.state1

print np.shape(model.model_time)
print np.shape(velocity)
print np.shape(friction)
print np.shape(state)

t = np.arange(0,30,0.01)

# Read in Chris Marone's Solutions
cjm_disp, cjm_mu = np.loadtxt('cjm_model_comparison/cjm_model.dis',skiprows=2,unpack=True)
cjm_time, cjm_mu = np.loadtxt('cjm_model_comparison/cjm_model.tim',skiprows=2,unpack=True)
cjm_state = np.loadtxt('cjm_model_comparison/cjm_model.sta',skiprows=2,unpack=True,usecols=[1])
cjm_vel = np.loadtxt('cjm_model_comparison/cjm_model.vel',skiprows=2,unpack=True,usecols=[0])
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
plt.plot(velocity,color='k',label='JRL')
plt.plot(cjm_vel,color='r',label='CJM')
#plt.plot(velocities,color='b')

fig = plt.figure(2)
plt.title('Friction')
plt.plot(model.model_time,friction,color='k',label='JRL')
plt.plot(cjm_time,cjm_mu,color='r',label='CJM')

fig = plt.figure(3)
plt.title('State Variable')
plt.plot(model.model_time,state,color='k',label='JRL')
plt.plot(cjm_time,cjm_state,color='r',label='CJM')

plt.show()
