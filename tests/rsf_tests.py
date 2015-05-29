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
        self.model.time = np.arange(0,40.01,0.01)
        lp_velocity = np.ones_like(self.model.time)
        lp_velocity[10*100:] = 10.
        self.model.loadpoint_velocity = lp_velocity
        self.model.solve()

    def test_max_friction(self):
        max_friction = np.max(self.model.results.friction)
        max_friction_index = np.argmax(self.model.results.friction)
        assert max_friction_index == 1129
        np.testing.assert_almost_equal(max_friction, 0.60784356, 8)

    def test_min_friction(self):
        min_friction = np.min(self.model.results.friction)
        min_friction_index = np.argmin(self.model.results.friction)
        assert min_friction_index == 1322
        np.testing.assert_almost_equal(min_friction, 0.58362349, 8)

    def test_max_state(self):
        max_state = np.max(self.model.results.states)
        max_state_index = np.argmax(self.model.results.states)
        assert max_state_index == 0
        np.testing.assert_almost_equal(max_state, 10.0, 8)

    def test_min_state(self):
        min_state = np.min(self.model.results.states)
        min_state_index = np.argmin(self.model.results.states)
        assert min_state_index == 1282
        np.testing.assert_almost_equal(min_state, 0.51710192, 8)

    def test_max_velocity(self):
        max_velocity = np.max(self.model.results.slider_velocity)
        max_velocity_index = np.argmax(self.model.results.slider_velocity)
        assert max_velocity_index == 1231
        np.testing.assert_almost_equal(max_velocity, 35.26869330, 8)

    def test_min_velocity(self):
        min_velocity = np.min(self.model.results.slider_velocity)
        min_velocity_index = np.argmin(self.model.results.slider_velocity)
        assert min_velocity_index == 0
        np.testing.assert_almost_equal(min_velocity, 1.0, 8)

    def teardown(self):
        print "TEAR DOWN!"
