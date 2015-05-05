import rsf
import numpy as np
import matplotlib.pyplot as plt
import sys

def error():
    print "Error! Valid test types are step and shs"
    sys.exit()

if len(sys.argv)<2:
    error()

test = sys.argv[1]

if test=='step':

    model = rsf.RateState()

    # Set model initial conditions
    model.mu0 = 0.6
    model.a = 0.005
    model.b = 0.01
    model.dc = 10.
    model.k = 1e-3
    model.v = 1.
    model.vlp = 10.
    model.vref = 1.

    # We want to solve for 30 seconds at 100Hz
    model.model_time = np.arange(0,40.01,0.01)

    # We want to slide at 1 um/s for 10 s, then at 10 um/s for 31
    lp_velocity = np.ones_like(model.model_time)
    lp_velocity[10*100:] = 10. # duplicate chris' step

    # Verify that things occur when they should
    #for t,v in zip(model.model_time,lp_velocity):
    #    print "%8.2f,\t%8.2f" %(t,v)

    # This makes an interesting case for sure. Different linear acceleartion profiles
    #lp_velocity[10*100:11*100] = np.linspace(1,10,100) # ramp over 1 second
    #lp_velocity[10*100:15*100] = np.linspace(1,10,500) # ramp over 5 seconds

    # Set the model load point velocity, must be same shape as model.model_time
    model.loadpoint_velocity = lp_velocity

    results = model.solve()

    velocity = results.slider_velocity
    friction = results.friction
    state = results.state1

    print np.shape(model.model_time)
    print np.shape(velocity)
    print np.shape(friction)
    print np.shape(state)

    t = np.arange(0,30,0.01)

    # Read in Chris Marone's Solutions
    cjm_disp, cjm_mud = np.loadtxt('cjm_model_comparison/cjm_model.dis',skiprows=2,unpack=True)
    cjm_time, cjm_mut = np.loadtxt('cjm_model_comparison/cjm_model.tim',skiprows=2,unpack=True)
    cjm_state = np.loadtxt('cjm_model_comparison/cjm_model.sta',skiprows=2,unpack=True,usecols=[1])
    cjm_vel,cjm_mu2 = np.loadtxt('cjm_model_comparison/cjm_model.vel',skiprows=2,unpack=True)
    cjm_vel = np.exp(cjm_vel)*1.

    cjm_disp[0] = -10.
    fig = plt.figure(1)
    plt.title('Slider Velocity')
    plt.plot(velocity,color='k',label='JRL')
    plt.plot(cjm_vel,color='r',label='CJM')
    #plt.plot(velocities,color='b')

    fig = plt.figure(2)
    plt.title('Friction')
    plt.plot(model.model_time,friction,color='k',label='JRL')
    plt.plot(cjm_time+10.,cjm_mut,color='r',label='CJM')

    fig = plt.figure(3)
    plt.title('State Variable')
    plt.plot(model.model_time,state,color='k',label='JRL')
    plt.plot(cjm_time+10.,cjm_state,color='r',label='CJM')

    fig = plt.figure(4)
    plt.title('Phase')
    #ln(v/v0),mu
    plt.plot(np.log(model.results.slider_velocity/model.vref),model.results.friction,color='k',label='JRL')
    plt.plot(np.log(cjm_vel/1.),cjm_mu2,color='r',label='CJM')

    fig = plt.figure(5)
    plt.title('Loading Velocity Profile')
    plt.plot(model.model_time, model.loadpoint_velocity)

    fig = plt.figure(6)
    plt.title('Time Displacement')
    plt.plot(model.model_time, model.results.displacement,color='k')

    fig = plt.figure(7)
    plt.title('Friction Displacement')
    plt.plot(model.results.displacement,friction,color='k',label='JRL')
    plt.plot(cjm_disp+10.,cjm_mud,color='r',label='CJM')

    plt.show()


if test == 'shs':
    ## SHS EXAMPLE

    model = rsf.RateState()

    # Set model initial conditions
    model.mu0 = 0.6
    model.a = 0.005
    model.b = 0.01
    model.dc = 10.
    model.k = 1e-3
    model.v = 1.
    model.vlp = 10.
    model.vref = 1.

    # We want to solve for 30 seconds at 100Hz
    model.model_time = np.arange(0,310.01,0.01)

    # We want to slide at 1 um/s for 10 s, then at 10 um/s for 31
    lp_velocity = np.ones_like(model.model_time)
    lp_velocity[10*100:110*100] = 0.0#1e-10 # duplicate chris' step

    # Set the model load point velocity, must be same shape as model.model_time
    model.loadpoint_velocity = lp_velocity

    results = model.solve()

    velocity = results.slider_velocity
    friction = results.friction
    state = results.state1

    print np.shape(model.model_time)
    print np.shape(velocity)
    print np.shape(friction)
    print np.shape(state)

    # Read in Chris Marone's Solutions
    cjm_disp, cjm_mu = np.loadtxt('cjm_model_comparison/cjm_shs.dis',skiprows=2,unpack=True)
    cjm_time, cjm_mu = np.loadtxt('cjm_model_comparison/cjm_shs.tim',skiprows=2,unpack=True)
    cjm_state = np.loadtxt('cjm_model_comparison/cjm_shs.sta',skiprows=2,unpack=True,usecols=[1])
    cjm_vel,cjm_mu2 = np.loadtxt('cjm_model_comparison/cjm_shs.vel',skiprows=2,unpack=True)
    cjm_vel = np.exp(cjm_vel)*1.

    fig = plt.figure(1)
    plt.title('Slider Velocity')
    plt.plot(velocity,color='k',label='JRL')
    plt.plot(cjm_vel,color='r',label='CJM')
    #plt.plot(velocities,color='b')

    fig = plt.figure(2)
    plt.title('Friction')
    plt.plot(model.model_time-10.,friction,color='k',label='JRL')
    plt.plot(cjm_time,cjm_mu,color='r',label='CJM')

    fig = plt.figure(3)
    plt.title('State Variable')
    plt.plot(model.model_time-10.,state,color='k',label='JRL')
    plt.plot(cjm_time,cjm_state,color='r',label='CJM')

    fig = plt.figure(4)
    plt.title('Phase')
    #ln(v/v0),mu
    plt.plot(np.log(model.results.slider_velocity/model.vref),model.results.friction,color='k',label='JRL')
    plt.plot(np.log(cjm_vel/1.),cjm_mu2,color='r',label='CJM')

    fig = plt.figure(5)
    plt.title('Loading Velocity Profile')
    plt.plot(model.model_time-10., model.loadpoint_velocity)


    plt.show()

if test == 'exp':
    ## SHS EXAMPLE

    model = rsf.RateState()

    # Set model initial conditions
    model.mu0 = 0.6
    model.a = 0.005
    model.b = 0.01
    model.dc = 10.
    model.k = 1e-3
    model.v = 1.
    model.vlp = 10.
    model.vref = 1.

    # We want to solve for 30 seconds at 100Hz
    model.model_time = np.arange(0,310.01,0.01)

    # We want to slide at 1 um/s for 10 s, then at 10 um/s for 31
    lp_velocity = np.ones_like(model.model_time)
    lp_velocity[8*100:10*100] = -1.
    lp_velocity[10*100:110*100] = 0.0#

    # Set the model load point velocity, must be same shape as model.model_time
    model.loadpoint_velocity = lp_velocity

    results = model.solve()

    velocity = results.slider_velocity
    friction = results.friction
    state = results.state1

    print np.shape(model.model_time)
    print np.shape(velocity)
    print np.shape(friction)
    print np.shape(state)

    # Read in Chris Marone's Solutions
    cjm_disp, cjm_mu = np.loadtxt('cjm_model_comparison/cjm_shs.dis',skiprows=2,unpack=True)
    cjm_time, cjm_mu = np.loadtxt('cjm_model_comparison/cjm_shs.tim',skiprows=2,unpack=True)
    cjm_state = np.loadtxt('cjm_model_comparison/cjm_shs.sta',skiprows=2,unpack=True,usecols=[1])
    cjm_vel,cjm_mu2 = np.loadtxt('cjm_model_comparison/cjm_shs.vel',skiprows=2,unpack=True)
    cjm_vel = np.exp(cjm_vel)*1.

    fig = plt.figure(1)
    plt.title('Slider Velocity')
    plt.plot(velocity,color='k',label='JRL')
    plt.plot(cjm_vel,color='r',label='CJM')
    #plt.plot(velocities,color='b')

    fig = plt.figure(2)
    plt.title('Friction')
    plt.plot(model.model_time-10.,friction,color='k',label='JRL')
    plt.plot(cjm_time,cjm_mu,color='r',label='CJM')

    fig = plt.figure(3)
    plt.title('State Variable')
    plt.plot(model.model_time-10.,state,color='k',label='JRL')
    plt.plot(cjm_time,cjm_state,color='r',label='CJM')

    fig = plt.figure(4)
    plt.title('Phase')
    #ln(v/v0),mu
    plt.plot(np.log(model.results.slider_velocity/model.vref),model.results.friction,color='k',label='JRL')
    plt.plot(np.log(cjm_vel/1.),cjm_mu2,color='r',label='CJM')

    fig = plt.figure(5)
    plt.title('Loading Velocity Profile')
    plt.plot(model.model_time-10., model.loadpoint_velocity)

    plt.show()

else:
    error()
