import numpy as np
from rsfmodel import rsf, staterelations, plotting

model = rsf.Model()

# Set model initial conditions
model.mu0 = 0.6 # Friction initial (at the reference velocity)
model.a = 0.005 # Empirical coefficient for the direct effect
model.k = 1e-3 # Normalized System stiffness (friction/micron)
model.v = 10. # Initial slider velocity, generally is vlp(t=0)
model.vref = 10. # Reference velocity, generally vlp(t=0)

state1 = staterelations.DieterichState()
state1.b = 0.01  # Empirical coefficient for the evolution effect
state1.Dc = 10.  # Critical slip distance

model.state_relations = [state1] # Which state relation we want to use

model.time = np.arange(0,150.01,0.01)

# We want to slide at 10 um/s for 20 s, hold for 100 s, then slide again
lp_velocity = 10* np.ones_like(model.time)
lp_velocity[20*100:120*100] = 0. # Velocity after 10 seconds is 10 um/s

# Set the model load point velocity, must be same shape as model.model_time
model.loadpoint_velocity = lp_velocity

# Run the model!
model.solve()

# Make the phase plot
plotting.phasePlot(model)

# Make a plot in displacement
plotting.dispPlot(model)

# Make a plot in time
plotting.timePlot(model)
