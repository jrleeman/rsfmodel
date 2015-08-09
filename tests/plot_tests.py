from nose.tools import *
import matplotlib
matplotlib.use('agg')
from rsfmodel import rsf, staterelations, plot
from rsfmodel.rsf import IncompleteModelError
import numpy as np

class TestDeiterichOneStateVar(object):

    def setup(self):
        self.model = rsf.Model()
        self.model.mu0 = 0.6
        self.model.a = 0.005
        self.model.k = 1e-3
        self.model.v = 1.
        self.model.vref = 1.
        state1 = staterelations.DieterichState()
        state1.b = 0.01
        state1.Dc = 10.
        self.model.state_relations = [state1]
        self.model.time = np.arange(0, 40.01, 1.)
        lp_velocity = np.ones_like(self.model.time)
        lp_velocity[10*1:] = 10.
        self.model.loadpoint_velocity = lp_velocity
        self.model.solve()

    def test_phaseplot(self):
        plot.phasePlot(self.model)

    @raises(ValueError)
    def test_phaseplot3D(self):
        plot.phasePlot3D(self.model)

    def test_dispplot(self):
        plot.dispPlot(self.model)

    def test_timeplot(self):
        plot.timePlot(self.model)


class TestDeiterichTwoStateVar(object):

    def setup(self):
        self.model = rsf.Model()
        self.model.mu0 = 0.6
        self.model.a = 0.012
        self.model.k = 3e-3
        self.model.v = 1.
        self.model.vref = 1.
        state1 = staterelations.DieterichState()
        state1.b = 0.0185
        state1.Dc = 5.

        state2 = staterelations.DieterichState()
        state2.b = 0.0088
        state2.Dc = 50.
        self.model.state_relations = [state1, state2]
        self.model.time = np.arange(0, 40.01, 1.)
        lp_velocity = np.ones_like(self.model.time)
        lp_velocity[10*1:] = 10.
        self.model.loadpoint_velocity = lp_velocity
        self.model.solve()

    def test_phaseplot(self):
        plot.phasePlot(self.model)

    def test_phaseplot3D(self):
        plot.phasePlot3D(self.model)

    def test_dispplot(self):
        plot.dispPlot(self.model)

    def test_timeplot(self):
        plot.timePlot(self.model)

    def test_friction(self):
        truth = np.array(
          [0.6, 0.6, 0.6, 0.6,
           0.6, 0.6, 0.6, 0.6,
           0.6, 0.6, 0.6, 0.61777816,
           0.56719338, 0.57926411, 0.58134415, 0.5741353,
           0.57495495, 0.57393578, 0.57211002, 0.5714098,
           0.57046462, 0.56953958, 0.56884951, 0.56819716,
           0.56762505, 0.56715188, 0.56674236, 0.5663961,
           0.56610841, 0.56586767, 0.565668, 0.56550353,
           0.56536809, 0.56525698, 0.56516614, 0.56509194,
           0.56503146, 0.56498223, 0.5649422, 0.56490967,
           0.56488327])

        np.testing.assert_almost_equal(self.model.results.friction, truth, 8)

    def test_state1(self):
        truth = np.array(
          [5., 5., 5., 5.,
           5., 5., 5., 5.,
           5., 5., 5., 2.80081321,
           0.26257732, 0.67749062, 0.5128508, 0.43041648,
           0.506089, 0.47944593, 0.47554365, 0.48734539,
           0.4843392, 0.48564166, 0.48855901, 0.48933082,
           0.49072826, 0.4922033, 0.49326055, 0.49430172,
           0.4952396, 0.49601255, 0.49668913, 0.4972666,
           0.49774673, 0.49815004, 0.49848567, 0.49876233,
           0.4989905, 0.4991779, 0.4993312, 0.49945649,
           0.49955867])

        np.testing.assert_almost_equal(self.model.results.states[:, 0],
                                       truth, 8)

    def test_state2(self):
        truth = np.array(
          [50., 50., 50., 50.,
           50., 50., 50., 50.,
           50., 50., 50., 47.03349418,
           28.26143675, 26.02288006, 22.50536043, 18.45103796,
           16.09903009, 13.99322929, 12.21984746, 10.8626045,
           9.74109978, 8.83011044, 8.10068484, 7.50795792,
           7.02833114, 6.64119123, 6.32770427, 6.07410906,
           5.86906354, 5.70314217, 5.56890919, 5.46032285,
           5.37246346, 5.30137737, 5.243863, 5.1973259,
           5.1596709, 5.12920263, 5.1045488, 5.08459968,
           5.06845739])

        np.testing.assert_almost_equal(self.model.results.states[:, 1],
                                       truth, 8)

    def test_slider_velocity(self):
        truth = np.array(
          [1., 1., 1., 1.,
           1., 1., 1., 1.,
           1., 1., 1., 11.24366669,
           9.27431239, 6.24880306, 12.6979352, 10.55410214,
           9.72918973, 10.76583201, 10.34192569, 10.2410124,
           10.3510415, 10.25600829, 10.22005129, 10.20908552,
           10.17177535, 10.1462882, 10.126265, 10.10491673,
           10.08761385, 10.07313912, 10.0603132, 10.04968798,
           10.04085724, 10.03343832, 10.02732652, 10.02229955,
           10.01816004, 10.01477376, 10.01200822, 10.00975089,
           10.00791305])

        np.testing.assert_almost_equal(self.model.results.slider_velocity,
                                       truth, 8)

    def test_time(self):
        truth = np.array(
          [0., 1., 2., 3.,
           4., 5., 6., 7.,
           8., 9., 10., 11.,
           12., 13., 14., 15.,
           16., 17., 18., 19.,
           20., 21., 22., 23.,
           24., 25., 26., 27.,
           28., 29., 30., 31.,
           32., 33., 34., 35.,
           36., 37., 38., 39.,
           40.])

        np.testing.assert_almost_equal(self.model.results.time, truth, 8)

    def test_loadpoint_displacement(self):
        truth = np.array(
          [0., 1., 2., 3.,
           4., 5., 6., 7.,
           8., 9., 10., 20.,
           30., 40., 50., 60.,
           70., 80., 90., 100.,
           110., 120., 130., 140.,
           150., 160., 170., 180.,
           190., 200., 210., 220.,
           230., 240., 250., 260.,
           270., 280., 290., 300.,
           310.])

        np.testing.assert_almost_equal(self.model.results.
                                       loadpoint_displacement, truth, 8)
