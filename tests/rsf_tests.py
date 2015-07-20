from nose.tools import *
import matplotlib
matplotlib.use('agg')
from rsfmodel import rsf
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
        state1 = rsf.DieterichState()
        state1.b = 0.01
        state1.Dc = 10.
        self.model.state_relations = [state1]
        self.model.time = np.arange(0,40.01,1.)
        lp_velocity = np.ones_like(self.model.time)
        lp_velocity[10*1:] = 10.
        self.model.loadpoint_velocity = lp_velocity
        self.model.solve()

    def test_phaseplot(self):
        rsf.phasePlot(self.model)

    @raises(ValueError)
    def test_phaseplot3D(self):
        rsf.phasePlot3D(self.model)

    def test_dispplot(self):
        rsf.dispPlot(self.model)

    def test_timeplot(self):
        rsf.timePlot(self.model)

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

    def test_slider_velocity(self):
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

        np.testing.assert_almost_equal(self.model.results.loadpoint_displacement, truth, 8)


class TestRuinaOneStateVar(object):

    def setup(self):
        self.model = rsf.Model()
        self.model.mu0 = 0.6
        self.model.a = 0.01
        self.model.k = 1e-3
        self.model.v = 1.
        self.model.vref = 1.
        state1 = rsf.RuinaState()
        state1.b = 0.005
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
           0.6, 0.6, 0.6, 0.60839299,
           0.6146288, 0.61663335, 0.61451424, 0.61239234,
           0.61152559, 0.61133979, 0.61138186, 0.61145348,
           0.61149781, 0.61151494, 0.61151781, 0.61151612,
           0.61151421, 0.61151317, 0.61151282, 0.61151279,
           0.61151285, 0.6115129, 0.61151292, 0.61151293,
           0.61151293, 0.61151293, 0.61151293, 0.61151293,
           0.61151293, 0.61151293, 0.61151293, 0.61151293,
           0.61151293])

        np.testing.assert_almost_equal(self.model.results.friction, truth,8)

    def test_state(self):
        truth = np.array(
          [10., 10., 10., 10.,
           10., 10., 10., 10.,
           10., 10., 10., 9.26098513,
           6.15593237, 2.47810827, 1.1310728, 0.91285565,
           0.92248181, 0.96185653, 0.98881543, 1.00027401,
           1.00273927, 1.00198602, 1.00086923, 1.00020489,
           0.99995887, 0.99992364, 0.99995225, 0.99998144,
           0.99999682, 1.00000177, 1.00000203, 1.00000112,
           1.00000038, 1.00000003, 0.99999994, 0.99999995,
           0.99999997, 0.99999999, 1., 1., 1.])

        np.testing.assert_almost_equal(self.model.results.states, truth.reshape((41,1)), 8)

    def test_slider_velocity(self):
        truth = np.array(
          [1., 1., 1., 1.,
           1., 1., 1., 1.,
           1., 1., 1., 2.40532918,
           5.50394141, 10.60027612, 12.69404325, 11.42856179,
           10.42488359, 10.02133885, 9.92544961, 9.93937282,
           9.97125078, 9.99210081, 10.00053876, 10.00216595,
           10.00149419, 10.00063054, 10.00013668, 9.99996121,
           9.99994058, 9.99996436, 9.99998669, 9.99999803,
           10.0000015, 10.00000156, 10.00000083, 10.00000027,
           10.00000002, 9.99999995, 9.99999996, 9.99999998,
           9.99999999])

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

        np.testing.assert_almost_equal(self.model.results.loadpoint_displacement, truth, 8)


class TestPerrinOneStateVar(object):

    def setup(self):
        self.model = rsf.Model()
        self.model.mu0 = 0.6
        self.model.a = 0.01
        self.model.k = 1e-3
        self.model.v = 1.
        self.model.vref = 1.
        state1 = rsf.PrzState()
        state1.b = 0.005
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
           0.6, 0.6, 0.6, 0.60839207,
           0.61456074, 0.61610039, 0.61386731, 0.61210792,
           0.61145825, 0.61134801, 0.61140235, 0.61146702,
           0.61150337, 0.61151606, 0.6115174, 0.61151556,
           0.61151389, 0.61151306, 0.61151281, 0.61151281,
           0.61151286, 0.61151291, 0.61151292, 0.61151293,
           0.61151293, 0.61151293, 0.61151293, 0.61151293,
           0.61151293, 0.61151293, 0.61151293, 0.61151293,
           0.61151293])

        np.testing.assert_almost_equal(self.model.results.friction, truth,8)

    def test_state(self):
        truth = np.array(
          [20., 20., 20., 20.,
           20., 20., 20., 20.,
           20., 20., 20., 18.43306111,
           11.27594869, 4.02202969, 2.08589836, 1.82377319,
           1.86690475, 1.93990872, 1.98492677, 2.00233846,
           2.0051588, 2.00332919, 2.00133051, 2.00024915,
           1.99988898, 1.99986167, 1.99992165, 1.99997245,
           1.99999695, 2.00000387, 2.00000357, 2.0000018,
           2.00000054, 2., 1.99999988, 1.99999991,
           1.99999996, 1.99999999, 2., 2.,
           2.])

        np.testing.assert_almost_equal(self.model.results.states, truth.reshape((41,1)), 8)

    def test_slider_velocity(self):
        truth = np.array(
          [1., 1., 1., 1.,
           1., 1., 1., 1.,
           1., 1., 1., 2.41090004,
           5.71219864, 11.15638292, 12.3913343, 11.11398288,
           10.29388702, 9.98762095, 9.92751362, 9.94838881,
           9.97758477, 9.99481799, 10.00114731, 10.00201236,
           10.00124394, 10.00047829, 10.00007902, 9.99995219,
           9.99994673, 9.99997095, 9.99999023, 9.9999992,
           10.00000158, 10.00000136, 10.00000066, 10.00000019,
           9.99999999, 9.99999995, 9.99999997, 9.99999999,
           10.])

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

        np.testing.assert_almost_equal(self.model.results.loadpoint_displacement, truth, 8)

class TestDeiterichTwoStateVar(object):

    def setup(self):
        self.model = rsf.Model()
        self.model.mu0 = 0.6
        self.model.a = 0.012
        self.model.k = 3e-3
        self.model.v = 1.
        self.model.vref = 1.
        state1 = rsf.DieterichState()
        state1.b = 0.0185
        state1.Dc = 5.

        state2 = rsf.DieterichState()
        state2.b = 0.0088
        state2.Dc = 50.
        self.model.state_relations = [state1, state2]
        self.model.time = np.arange(0,40.01,1.)
        lp_velocity = np.ones_like(self.model.time)
        lp_velocity[10*1:] = 10.
        self.model.loadpoint_velocity = lp_velocity
        self.model.solve()

    def test_phaseplot(self):
        rsf.phasePlot(self.model)

    def test_phaseplot3D(self):
        rsf.phasePlot3D(self.model)

    def test_dispplot(self):
        rsf.dispPlot(self.model)

    def test_timeplot(self):
        rsf.timePlot(self.model)

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

        np.testing.assert_almost_equal(self.model.results.friction, truth,8)

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

        np.testing.assert_almost_equal(self.model.results.states[:,0], truth, 8)

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

        np.testing.assert_almost_equal(self.model.results.states[:,1], truth, 8)

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

        np.testing.assert_almost_equal(self.model.results.loadpoint_displacement, truth, 8)

class TestRuinaTwoStateVar(object):

    def setup(self):
        self.model = rsf.Model()
        self.model.mu0 = 0.6
        self.model.a = 0.012
        self.model.k = 8e-3
        self.model.v = 1.
        self.model.vref = 1.
        state1 = rsf.RuinaState()
        state1.b = 0.0185
        state1.Dc = 5.

        state2 = rsf.RuinaState()
        state2.b = 0.0088
        state2.Dc = 50.
        self.model.state_relations = [state1, state2]
        self.model.time = np.arange(0,40.01,1.)
        lp_velocity = np.ones_like(self.model.time)
        lp_velocity[10*1:] = 10.
        self.model.loadpoint_velocity = lp_velocity
        self.model.solve()

    def test_friction(self):
        truth = np.array(
          [0.6, 0.6, 0.6, 0.6,
           0.6, 0.6, 0.6, 0.6,
           0.6, 0.6, 0.6, 0.57317487,
           0.57620828, 0.57423257, 0.57241419, 0.57098114,
           0.56981792, 0.56887496, 0.56810967, 0.56748807,
           0.56698283, 0.56657196, 0.56623767, 0.56596561,
           0.56574412, 0.56556376, 0.56541686, 0.5652972,
           0.56519972, 0.56512029, 0.56505557, 0.56500283,
           0.56495985, 0.56492482, 0.56489628, 0.56487301,
           0.56485404, 0.56483859, 0.56482599, 0.56481572,
           0.56480735])

        np.testing.assert_almost_equal(self.model.results.friction, truth,8)

    def test_state1(self):
        truth = np.array(
          [5., 5., 5., 5.,
           5., 5., 5., 5.,
           5., 5., 5., 0.34503501,
           0.48718318, 0.48688665, 0.48906973, 0.49116621,
           0.49281743, 0.49415943, 0.49524843, 0.49613296,
           0.4968519, 0.49743656, 0.49791223, 0.49829936,
           0.49861453, 0.49887118, 0.4990802, 0.49925047,
           0.49938918, 0.4995022, 0.49959429, 0.49966934,
           0.49973049, 0.49978034, 0.49982096, 0.49985406,
           0.49988105, 0.49990304, 0.49992097, 0.49993558,
           0.49994749])

        np.testing.assert_almost_equal(self.model.results.states[:,0], truth, 8)

    def test_state2(self):
        truth = np.array(
          [50., 50., 50., 50.,
           50., 50., 50., 50.,
           50., 50., 50., 25.46871646,
           19.28091149, 14.94589606, 12.15504322, 10.28679987,
           8.98658279, 8.05413768, 7.36902545, 6.85569405,
           6.46493829, 6.16365308, 5.92892636, 5.74450566,
           5.59861356, 5.48255637, 5.38981385, 5.31542877,
           5.2555883, 5.20733084, 5.1683369, 5.13677719,
           5.11120066, 5.09045071, 5.07360177, 5.05991069,
           5.04877914, 5.03972433, 5.03235598, 5.0263581,
           5.02147453])

        np.testing.assert_almost_equal(self.model.results.states[:,1], truth, 8)

    def test_slider_velocity(self):
        truth = np.array(
          [1., 1., 1., 1.,
           1., 1., 1., 1.,
           1., 1., 1., 10.81546891,
           10.03428877, 10.2683735, 10.19823177, 10.16123603,
           10.13064517, 10.10597718, 10.08604871, 10.06991893,
           10.05684658, 10.04624063, 10.03762818, 10.03062956,
           10.02493909, 10.02031007, 10.01654309, 10.01347665,
           10.01097985, 10.00894645, 10.00729017, 10.00594089,
           10.00484157, 10.00394584, 10.00321593, 10.00262111,
           10.00213635, 10.00174128, 10.00141929, 10.00115686,
           10.00094295])

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

        np.testing.assert_almost_equal(self.model.results.loadpoint_displacement, truth, 8)

class TestPRZOneStateVar(object):

    def setup(self):
        self.model = rsf.Model()
        self.model.mu0 = 0.6
        self.model.a = 0.011
        self.model.k = 3e-3
        self.model.v = 1.
        self.model.vref = 1.
        state1 = rsf.PrzState()
        state1.b = 0.01
        state1.Dc = 5.
        self.model.state_relations = [state1]
        self.model.time = np.arange(0,40.01,1)
        lp_velocity = np.ones_like(self.model.time)
        lp_velocity[10*1:] = 10.
        self.model.loadpoint_velocity = lp_velocity
        self.model.solve()

    def test_friction(self):
        truth = np.array(
          [0.6, 0.6, 0.6, 0.6,
           0.6, 0.6, 0.6, 0.6,
           0.6, 0.6, 0.6, 0.61505335,
            0.60008169, 0.60202942, 0.60245015, 0.60230027,
            0.60229482, 0.60230363, 0.60230288, 0.60230249,
            0.60230258, 0.60230259, 0.60230258, 0.60230258,
            0.60230259, 0.60230259, 0.60230259, 0.60230259,
            0.60230259, 0.60230259, 0.60230259, 0.60230259,
            0.60230259, 0.60230259, 0.60230259, 0.60230259,
            0.60230259, 0.60230259, 0.60230259, 0.60230259,
            0.60230259])

        np.testing.assert_almost_equal(self.model.results.friction, truth,8)

    def test_state(self):
        truth = np.array(
          [10., 10., 10., 10.,
        10., 10., 10., 10.,
        10., 10., 10., 2.09747593,
        0.81845159, 1.02623769, 1.00892332, 0.99754925,
        0.99981141, 1.00015612, 0.99999181, 0.99999246,
        1.00000134, 1.00000025, 0.9999999, 1.,
        1.00000001, 1., 1., 1.,
        1., 1., 1., 1.,
        1., 1., 1., 1.,
        1., 1., 1., 1.,
        1.])

        np.testing.assert_almost_equal(self.model.results.states, truth.reshape((41,1)), 8)

    def test_slider_velocity(self):
        truth = np.array(
          [1., 1., 1., 1.,
        1., 1., 1., 1.,
        1., 1., 1., 16.25418209,
        9.80422815, 9.52773797, 10.05352858, 10.02022142,
        9.99465394, 9.99953164, 10.00034701, 9.99998443,
        9.99998292, 10.00000287, 10.00000059, 9.99999977,
        9.99999999, 10.00000001, 10., 10.,
        10., 10., 10., 10.,
        10., 10., 10., 10.,
        10., 10., 10., 10.,
        10.])

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

        np.testing.assert_almost_equal(self.model.results.loadpoint_displacement, truth, 8)


class TestPRZTwoStateVar(object):

    def setup(self):
        self.model = rsf.Model()
        self.model.mu0 = 0.6
        self.model.a = 0.02
        self.model.k = 3e-3
        self.model.v = 1.
        self.model.vref = 1.
        state1 = rsf.PrzState()
        state1.b = 0.01
        state1.Dc = 5.

        state2 = rsf.PrzState()
        state2.b = 0.005
        state2.Dc = 3.
        self.model.state_relations = [state1, state2]
        self.model.time = np.arange(0,40.01,1.)
        lp_velocity = np.ones_like(self.model.time)
        lp_velocity[10*1:] = 10.
        self.model.loadpoint_velocity = lp_velocity
        self.model.solve()

    def test_friction(self):
        truth = np.array(
          [0.6, 0.6, 0.6, 0.6,
        0.6, 0.6, 0.6, 0.6,
        0.6, 0.6, 0.6, 0.62332749,
        0.61662392, 0.60925037, 0.61102238, 0.61175384,
        0.61157669, 0.61148672, 0.611505, 0.61151574,
        0.6115139, 0.61151263, 0.61151281, 0.61151296,
        0.61151294, 0.61151292, 0.61151292, 0.61151293,
        0.61151293, 0.61151293, 0.61151293, 0.61151293,
        0.61151293, 0.61151293, 0.61151293, 0.61151293,
        0.61151293, 0.61151293, 0.61151293, 0.61151293,
        0.61151293])

        np.testing.assert_almost_equal(self.model.results.friction, truth,8)

    def test_state1(self):
        truth = np.array(
          [10., 10., 10., 10.,
        10., 10., 10., 10.,
        10., 10., 10., 6.84431098,
        0.72378553, 0.86994234, 1.03299015, 1.01865937,
        0.99670972, 0.99782031, 1.00032473, 1.00025833,
        0.99996907, 0.99996963, 1.00000281, 1.00000355,
        0.99999976, 0.99999959, 1.00000002, 1.00000005,
        1., 0.99999999, 1., 1.,
        1., 1., 1., 1.,
        1., 1., 1., 1.,
        1.])

        np.testing.assert_almost_equal(self.model.results.states[:,0], truth, 8)

    def test_state2(self):
        truth = np.array(
          [6., 6., 6., 6.,
        6., 6., 6., 6.,
        6., 6., 6., 3.4500867,
        0.36317378, 0.55303551, 0.63530577, 0.60831617,
        0.59630691, 0.59898905, 0.60040086, 0.60012611,
        0.59995714, 0.5999845, 0.60000454, 0.60000189,
        0.59999953, 0.59999977, 0.60000005, 0.60000003,
        0.6, 0.6, 0.6, 0.6,
        0.6, 0.6, 0.6, 0.6,
        0.6, 0.6, 0.6, 0.6,
        0.6])

        np.testing.assert_almost_equal(self.model.results.states[:,1], truth, 8)

    def test_slider_velocity(self):
        truth = np.array(
          [1., 1., 1., 1.,
        1., 1., 1., 1.,
        1., 1., 1., 4.45620908,
        17.20632436, 9.77177682, 9.46437404, 9.99361157,
        10.06399985, 10.00202131, 9.99274721, 9.99958814,
        10.000822, 10.0000672, 9.99990739, 9.99999004,
        10.00001038, 10.00000139, 9.99999884, 9.99999981,
        10.00000013, 10.00000002, 9.99999999, 10.,
        10., 10., 10., 10.,
        10., 10., 10., 10.,
        10.])

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

        np.testing.assert_almost_equal(self.model.results.loadpoint_displacement, truth, 8)


class TestRuinaTwoStateVarMissing(object):

    def setup(self):
        self.model = rsf.Model()
        self.model.mu0 = 0.6
        self.model.a = 0.012
        self.model.k = 8e-3
        self.model.v = 1.
        self.model.vref = 1.
        state1 = rsf.RuinaState()
        state1.b = 0.0185
        state1.Dc = 5.

        state2 = rsf.RuinaState()
        state2.b = 0.0088
        state2.Dc = 50.
        self.model.state_relations = [state1, state2]
        self.model.time = np.arange(0,40.01,1.)
        lp_velocity = np.ones_like(self.model.time)
        lp_velocity[10*1:] = 10.
        self.model.loadpoint_velocity = lp_velocity
        self.model.solve()

    @raises(IncompleteModelError)
    def test_a_missing(self):
        self.model.a = None
        self.model.solve()

    @raises(IncompleteModelError)
    def test_vref_missing(self):
        self.model.vref = None
        self.model.solve()

    @raises(Exception)
    def test_state_realtions_missing(self):
        self.model.state_relations = []
        self.model.solve()

    @raises(IncompleteModelError)
    def test_k_missing(self):
        self.model.k = None
        self.model.solve()

    @raises(IncompleteModelError)
    def test_time_missing(self):
        self.model.time = None
        self.model.solve()

    @raises(IncompleteModelError)
    def test_lp_velocity_missing(self):
        self.model.loadpoint_velocity = None
        self.model.solve()

    @raises(IncompleteModelError)
    def test_state1_b_missing(self):
        state1 = rsf.RuinaState()
        state1.Dc = 5.

        state2 = rsf.RuinaState()
        state2.b = 0.0088
        state2.Dc = 50.
        self.model.state_relations = [state1, state2]
        self.model.solve()

    @raises(IncompleteModelError)
    def test_state2_b_missing(self):
        state1 = rsf.RuinaState()
        state1.b = 0.0185
        state1.Dc = 5.

        state2 = rsf.RuinaState()
        state2.Dc = 50.
        self.model.state_relations = [state1, state2]
        self.model.solve()

    @raises(IncompleteModelError)
    def test_state1_Dc_missing(self):
        state1 = rsf.RuinaState()
        state1.b = 0.0185

        state2 = rsf.RuinaState()
        state2.b = 0.0088
        state2.Dc = 50.
        self.model.state_relations = [state1, state2]
        self.model.solve()

    @raises(IncompleteModelError)
    def test_state2Dcb_missing(self):
        state1 = rsf.RuinaState()
        state1.b = 0.0185
        state1.Dc = 5.

        state2 = rsf.RuinaState()
        state2.b = 0.0088
        self.model.state_relations = [state1, state2]
        self.model.solve()

    @raises(IncompleteModelError)
    def test_time_velocity_length_mismatch(self):
        self.model.time = np.arange(0,40.01,0.1)
        self.model.solve()

class TestNagataOneStateVar(object):

    def setup(self):
        self.model = rsf.Model()
        self.model.mu0 = 0.6
        self.model.a = 0.027
        self.model.k = 0.01
        self.model.v = 1.
        self.model.vref = 1.
        state1 = rsf.NagataState()
        state1.b = 0.029
        state1.Dc = 3.33
        state1.c = 2.
        self.model.state_relations = [state1]
        self.model.time = np.arange(0,41,1)
        lp_velocity = np.ones_like(self.model.time)
        lp_velocity[10*1:] = 10.
        self.model.loadpoint_velocity = lp_velocity
        self.model.solve(hmax=0.001,mxstep=5000)

    def test_phaseplot(self):
        rsf.phasePlot(self.model)

    @raises(ValueError)
    def test_phaseplot3D(self):
        rsf.phasePlot3D(self.model)

    def test_dispplot(self):
        rsf.dispPlot(self.model)

    def test_timeplot(self):
        rsf.timePlot(self.model)

    def test_friction(self):
        truth = np.array(
          [0.6, 0.6, 0.6, 0.6,
            0.6, 0.6, 0.6, 0.6,
            0.6, 0.6, 0.6, 0.63405624,
            0.59077486, 0.59622046, 0.5952689, 0.59541182,
            0.59539301, 0.59539492, 0.59539486, 0.59539482,
            0.59539483, 0.59539483, 0.59539483, 0.59539483,
            0.59539483, 0.59539483, 0.59539483, 0.59539483,
            0.59539483, 0.59539483, 0.59539483, 0.59539483,
            0.59539483, 0.59539483, 0.59539483, 0.59539483,
            0.59539483, 0.59539483, 0.59539483, 0.59539483,
            0.59539483])

        np.testing.assert_almost_equal(self.model.results.friction, truth,8)

    def test_state(self):
        truth = np.array(
          [3.33, 3.33, 3.33, 3.33,
        3.33, 3.33, 3.33, 3.33,
        3.33, 3.33, 3.33, 0.75351918,
        0.30973464, 0.34020859, 0.33152619, 0.33327998,
        0.33295162, 0.33300762, 0.33299893, 0.33300012,
        0.33299999, 0.333, 0.333, 0.333,
        0.333, 0.333, 0.333, 0.333,
        0.333, 0.333, 0.333, 0.333,
        0.333, 0.333, 0.333, 0.333,
        0.333, 0.333, 0.333, 0.333,
        0.333])

        np.testing.assert_almost_equal(self.model.results.states, truth.reshape((41,1)), 8)

    def test_slider_velocity(self):
        truth = np.array(
          [1., 1., 1., 1.,
        1., 1., 1., 1.,
        1., 1., 1., 17.4159695,
        9.10903394, 10.07604967, 10.00100091, 9.99726612,
        10.00088537, 9.99978713, 10.00004417, 9.99999173,
        10.00000142, 9.99999978, 10.00000003, 10.,
        10., 10., 10., 10.,
        10., 10., 10., 10.,
        10., 10., 10., 10.,
        10., 10., 10., 10.,
        10.])

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

        np.testing.assert_almost_equal(self.model.results.loadpoint_displacement, truth, 8)