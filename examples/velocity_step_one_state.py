import numpy as np
import matplotlib.pyplot as plt
from rsfmodel import rsf

model = rsf.Model()

# Set model initial conditions
model.mu0 = 0.6 # Friction initial (at the reference velocity)
model.a = 0.005 # Empirical coefficient for the direct effect
model.k = 1e-3 # Normalized System stiffness (friction/micron)
model.v = 1. # Initial slider velocity, generally is vlp(t=0)
model.vref = 1. # Reference velocity, generally vlp(t=0)

state1 = rsf.DieterichState()
state1.b = 0.01  # Empirical coefficient for the evolution effect
state1.Dc = 10.  # Critical slip distance

model.state_relations = [state1] # Which state relation we want to use

# We want to solve for 40 seconds at 100Hz
model.time = np.arange(0,40.01,0.01)

# We want to slide at 1 um/s for 10 s, then at 10 um/s for 31
lp_velocity = np.ones_like(model.time)
lp_velocity[10*100:] = 10. # Velocity after 10 seconds is 10 um/s

# Set the model load point velocity, must be same shape as model.model_time
model.loadpoint_velocity = lp_velocity

# Run the model!
model.solve()

# Make the phase plot
rsf.phasePlot(model)

# Make a plot in displacement
rsf.dispPlot(model)

# Make a plot in time
rsf.timePlot(model)
