from nose.tools import *
from rsfmodel import rsf
import numpy as np

class TestDeiterichOneStateVar(object):

    def setup(self):
        self.model = rsf.Model()
        self.model.mu0 = 0.6
        self.model.a = 0.005
        self.model.k = 1e-3
        self.model.v = 1.
        self.model.vref = 1.
        state1 = rsf.DieterichState(self.model)
        state1.b = 0.01
        state1.Dc = 10.
        self.model.state_relations = [state1]
        self.model.time = np.arange(0,40.01,1.)
        lp_velocity = np.ones_like(self.model.time)
        lp_velocity[10*1:] = 10.
        self.model.loadpoint_velocity = lp_velocity
        self.model.solve()

    def test_friction(self):
        truth = np.array(
          [0.6, 0.6, 0.6, 0.6,
           0.6, 0.6, 0.6, 0.6,
           0.6, 0.6, 0.6, 0.60720485,
           0.60186637, 0.5840091, 0.58563096, 0.58899543,
           0.58998186, 0.58879751, 0.58803412, 0.58825653,
           0.58858313, 0.58860319, 0.58848625, 0.58844431,
           0.58847472, 0.58849913, 0.58849522, 0.58848506,
           0.58848348, 0.58848674, 0.5884883, 0.58848756,
           0.58848677, 0.5884868, 0.58848711, 0.58848718,
           0.5884871, 0.58848704, 0.58848706, 0.58848708,
           0.58848708])

        np.testing.assert_almost_equal(self.model.results.friction, truth,8)

    def test_state(self):
        truth = np.array(
          [10., 10., 10., 10.,
           10., 10., 10., 10.,
           10., 10., 10., 8.39993716,
           2.2355129, 0.54248237, 0.94936884, 1.21163659,
           1.13491241, 0.96889358, 0.94896281, 0.99777122,
           1.01876822, 1.00674821, 0.99515136, 0.9961297,
           1.00063816, 1.00161995, 1.00024914, 0.9994787,
           0.99975352, 1.00011844, 1.00012598, 0.99999394,
           0.99995185, 0.9999879, 1.00001411, 1.00000865,
           0.99999738, 0.99999604, 0.99999978, 1.00000139,
           1.00000049])

        np.testing.assert_almost_equal(self.model.results.states, truth.reshape((41,1)), 8)

    def test_slider_velosity(self):
        truth = np.array(
          [1., 1., 1., 1.,
           1., 1., 1., 1.,
           1., 1., 1., 5.98760938,
           29.06408439, 13.87639287, 6.26688086, 7.54067754,
           10.46913416, 11.33475717, 10.14281987, 9.59208664,
           9.82182937, 10.0981912, 10.09601357, 9.99203098,
           9.9626088, 9.99173502, 10.01132361, 10.00641151,
           9.99774235, 9.99697087, 9.99992839, 10.00109331,
           10.00035175, 9.99970252, 9.99978205, 10.00004456,
           10.00009344, 10.00001142, 9.99996902, 9.99998658,
           10.0000074])

        np.testing.assert_almost_equal(self.model.results.slider_velocity, truth, 8)

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

    def test_displacement(self):
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

        np.testing.assert_almost_equal(self.model.results.loadpoint_displacement, truth, 8)
