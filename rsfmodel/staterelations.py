import numpy as np
from math import log

class StateRelation(object):
    """
    Abstract state relation object that contains the generally used atributes
    in state relations (b,Dc).
    """
    def __init__(self):
        self.b = None
        self.Dc = None
        self.state = None

    def velocity_component(self, system):
        """
        General velocity contribution from a given state variable

        .. math::
        V_\text{contribution} = b \text{ln}\left(\frac{V_0 \theta}{D_c}\right)
        """
        return self.b * np.log(system.vref * self.state / self.Dc)


class DieterichState(StateRelation):
    """
    The slowness or Dieterich state relation as proposed by Jim Dieterich (1979)


    .. math::
    \frac{d\theta}{dt} = 1 - \frac{V_\text{slider} \theta}{D_c}
    """
    def set_steady_state(self, system):
        self.state = self.Dc/system.vref

    def evolve_state(self, system):
        return 1. - system.v * self.state / self.Dc


class RuinaState(StateRelation):
    """
    The slip or Ruina state relation as proposed by Andy Ruina (1983)

    .. math::
    \frac{d\theta}{dt} =  -\frac{V_\text{slider} \theta}{D_c}
    \text{ln}\left(\frac{V_\text{slider} \theta}{D_c}\right)
    """
    def set_steady_state(self, system):
        self.state = self.Dc/system.vref

    def evolve_state(self, system):
        return -1 * (system.v * self.state / self.Dc) * log(system.v * self.state / self.Dc)


class PrzState(StateRelation):
    """
    The PRZ state relation as proposed by Perrin, Rice, and Zheng (1995):

    .. math::
    \frac{d\theta}{dt} =  1 - \left(\frac{V_\text{slider} \theta}{2D_c}\right) ^2
    """
    def set_steady_state(self, system):
        self.state = 2 * self.Dc / system.v
        self.prz_vref = system.vref/(2*self.Dc)

    def evolve_state(self, system):
        return 1. - (system.v * self.state / (2 * self.Dc))**2

    def velocity_component(self, system):
        """
        Perrin-Rice velocity contribution

        .. math::
        V_\text{contribution} = b \text{ln}\left(V_{\text{prz}0} \theta\right)
        """
        return self.b * np.log(self.prz_vref * self.state)


class NagataState(StateRelation):
    """
    The Nagata state relation as proposed by Nagata et al. (2012):

    .. math::
    \frac{d\theta}{dt} =  1 - \frac{V_\text{slider} \theta}{D_c}
    - \frac{c}{b}\theta\frac{d\mu}{dt}
    """
    def __init__(self):
        StateRelation.__init__(self)
        self.c = None

    def set_steady_state(self, system):
        self.state = self.Dc / system.vref

    def evolve_state(self, system):
        return 1. - (system.v * self.state / self.Dc) - \
               (self.c / self.b * self.state * system.dmu_dt)
