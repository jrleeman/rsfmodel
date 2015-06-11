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
        state1 = rsf.RuinaState(self.model)
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
        state1 = rsf.PrzState(self.model)
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
        state1 = rsf.DieterichState(self.model)
        state1.b = 0.0185
        state1.Dc = 5.

        state2 = rsf.DieterichState(self.model)
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
