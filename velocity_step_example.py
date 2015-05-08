import numpy as np
import matplotlib.pyplot as plt
import rsf

model = rsf.RateState()

# Set model initial conditions
model.mu0 = 0.6 # Friction initial (at the reference velocity)
model.a = 0.005 # Empirical coefficient for the direct effect
model.b = 0.01 # Empirical coefficient for the evolution effect
model.dc = 10. # Critical slip distance
model.k = 1e-3 # Normalized System stiffness (friction/micron)
model.v = 1. # Initial slider velocity, generally is vlp(t=0)
model.vref = 1. # Reference velocity, generally vlp(t=0)
model.stateLaw = model.dieterichState # Which state relation we want to use

# We want to solve for 40 seconds at 100Hz
model.model_time = np.arange(0,40.01,0.01)

# We want to slide at 1 um/s for 10 s, then at 10 um/s for 31
lp_velocity = np.ones_like(model.model_time)
lp_velocity[10*100:] = 10. # Velocity after 10 seconds is 10 um/s

# Set the model load point velocity, must be same shape as model.model_time
model.loadpoint_velocity = lp_velocity

# Run the model!
results = model.solve()

# Make the phase plot
model.phasePlot()

# Make a plot in displacement
model.dispPlot()

# Make a plot in time
model.timePlot()
