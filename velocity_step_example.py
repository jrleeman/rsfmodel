import numpy as np
import matplotlib.pyplot as plt
import rsf

model = rsf.RateState()

# Set model initial conditions
model.mu0 = 0.6
model.a = 0.005
model.b = 0.01
model.dc = 10.
model.k = 1e-3
#model.k = 2.5e-3
model.v = 1.
model.vlp = 10.
model.vref = 1.
model.stateLaw = model.dieterichState

# We want to solve for 30 seconds at 100Hz
model.model_time = np.arange(0,40.01,0.01)

# We want to slide at 1 um/s for 10 s, then at 10 um/s for 31
lp_velocity = np.ones_like(model.model_time)
lp_velocity[10*100:] = 10. # duplicate chris' step

# Set the model load point velocity, must be same shape as model.model_time
model.loadpoint_velocity = lp_velocity

results = model.solve()

# Make the phase plot
model.phasePlot()

# Make a couple of stanard plots
fig = plt.figure()
ax1 = plt.subplot(111)
ax2 = ax1.twinx()

ax1.plot(model.results.time,model.results.friction,color='k')
ax2.plot(model.results.time,model.results.slider_velocity,color='b')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Friction')
ax2.set_ylabel('Slider Velocity')

fig = plt.figure()
ax1 = plt.subplot(111)
ax2 = ax1.twinx()

ax1.plot(model.results.displacement,model.results.friction,color='k')
ax2.plot(model.results.displacement,model.results.state1,color='g')
ax1.set_xlabel('Displacement [um]')
ax1.set_ylabel('Friction')
ax2.set_ylabel('State Variable')
plt.show()
