import numpy as np
import matplotlib.pyplot as plt
from rsfmodel import rsf

model = rsf.Model()

# Set model initial conditions
model.mu0 = 0.6 # Friction initial (at the reference velocity)
model.a = 0.027 # Empirical coefficient for the direct effect
#model.a = 0.05
model.k = 0.01 # Normalized System stiffness (friction/micron)
model.v = 1. # Initial slider velocity, generally is vlp(t=0)
model.vref = 1. # Reference velocity, generally vlp(t=0)

#state1 = rsf.RuinaState()
state1 = rsf.NagataState()
state1.b = 0.029  # Empirical coefficient for the evolution effect
state1.Dc = 3.33  # Critical slip distance
state1.c = 2.

model.state_relations = [state1] # Which state relation we want to use

#slip,delta_tau,normal_stress,load_point,time,delta_mu,load_pt_vel,slip_vel,state = np.loadtxt('RSF_from_Path/stress_profN_1.out', unpack=True)


# We want to solve for 40 seconds at 100Hz

model.time = np.arange(0,40.01,0.01)
#model.time = time

# We want to slide at 1 um/s for 10 s, then at 10 um/s for 31
lp_velocity = np.ones_like(model.time)
lp_velocity[4.9*100:] = 10. # Velocity after 10 seconds is 10 um/s

# Set the model load point velocity, must be same shape as model.model_time
model.loadpoint_velocity = lp_velocity
#model.loadpoint_velocity = load_pt_vel

# Run the model!
model.solve(tcrit = model.time[480:600])
#model.solve()
#hmax =0.01
# Make the phase plot
#rsf.phasePlot(model)

# # Make a plot in displacement
#rsf.dispPlot(model)
#
# # Make a plot in time
#rsf.timePlot(model)

#4.9
#34
#326
#328.8
