import numpy as np
from math import log

class StateRelation(object):
    """
    Abstract state relation object that contains the generally used atributes
    in state relations (b,Dc).

    Attributes
    ----------
    b : float
        Rate and state empirical parameter b.
    Dc : float
        Critical slip distance.
    state : float
        State variable.
    """
    def __init__(self):
        self.b = None
        self.Dc = None
        self.state = None

    def velocity_component(self, system):
        """
        General velocity contribution from a given state variable

        Notes
        -----
        .. math::
            V_\\text{contribution} = b \\text{ln}\\left(\\frac{V_0 \\theta}{D_c}\\right)
        """
        return self.b * np.log(system.vref * self.state / self.Dc)


class DieterichState(StateRelation):
    """
    The slowness or Dieterich state relation as proposed by [#Dieterich1979]_.

    Notes
    -----
    .. math::
      \\frac{d\\theta}{dt} = 1 - \\frac{V_\\text{slider} \\theta}{D_c}

    .. [#Dieterich1979] Dieterich, J. "Modeling of rock friction: 1. Experimental
       results and constitutive equations." Journal of Geophysical
       Research: Solid Earth (19782012) 84.B5 (1979): 2161-2168.
    """
    def set_steady_state(self, system):
        self.state = self.Dc/system.vref

    def evolve_state(self, system):
        return 1. - system.v * self.state / self.Dc


class RuinaState(StateRelation):
    """
    The slip or Ruina state relation as proposed by [#Ruina1983]_.

    Notes
    -----
    .. math::
      \\frac{d\theta}{dt} =  -\\frac{V_\\text{slider} \\theta}{D_c}
      \\text{ln}\left(\\frac{V_\\text{slider} \\theta}{D_c}\\right)

    .. [#Ruina1983] Ruina, Andy. "Slip instability and state variable friction laws."
       J. geophys. Res 88.10 (1983): 359-10.
    """
    def set_steady_state(self, system):
        self.state = self.Dc/system.vref

    def evolve_state(self, system):
        return -1 * (system.v * self.state / self.Dc) * log(system.v * self.state / self.Dc)


class PrzState(StateRelation):
    """
    The PRZ state relation as proposed by [#PRZ1995]_:

    Notes
    -----
    .. math::
      \\frac{d\\theta}{dt} =  1 - \left(\\frac{V_\\text{slider} \\theta}{2D_c}\\right) ^2

    .. [#PRZ1995] Perrin, G., Rice, J., and Zheng, G.
       "Self-healing slip pulse on a frictional surface."
       Journal of the Mechanics and Physics of Solids 43.9 (1995): 1461-1495.
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
          V_\\text{contribution} = b \\text{ln}\left(V_{\\text{prz}0} \\theta\\right)
        """
        return self.b * np.log(self.prz_vref * self.state)


class NagataState(StateRelation):
    """
    The Nagata state relation as proposed by [#Nagata2012]_:

    Notes
    -----
    .. math::
      \\frac{d\\theta}{dt} =  1 - \\frac{V_\\text{slider} \\theta}{D_c}
      - \\frac{c}{b}\\theta\\frac{d\mu}{dt}

    .. [#Nagata2012] Nagata, K., Nakatani, M., Yoshida, S., "A revised rate-and-state
       -dependent friction law obtained by constraining constitutive and
       evolution laws separately with laboratory data," Journal of Geophysical
       Research: Solid Earth, vol 117, 2012.
    """
    def __init__(self):
        StateRelation.__init__(self)
        self.c = None

    def set_steady_state(self, system):
        self.state = self.Dc / system.vref

    def evolve_state(self, system):
        return 1. - (system.v * self.state / self.Dc) - \
               (self.c / self.b * self.state * system.dmu_dt)
