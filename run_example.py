import rsf
import numpy as np
import matplotlib.pyplot as plt

model = rsf.RateState()
results = model.solve()

velocity = results.velocity
friction = results.friction
state = results.state1

print np.shape(np.arange(0,30,0.01))
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
