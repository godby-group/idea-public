""" Tests for non-interacting systems

""" 
from __future__ import absolute_import
from . import NON
from . import input
import unittest

# decimal places for comparison of results
d = 6

class NONTestHarmonic(unittest.TestCase):
    """ Tests on the harmonic oscillator potential

    """ 

    def setUp(self):
        """ Sets up harmonic oscillator system """
        pm = input.Input()
        pm.run.LDA = True
        pm.run.save = False
        pm.run.verbosity = 'low'

        ### system parameters
        sys = pm.sys
        sys.NE = 2                  #: Number of electrons
        sys.grid = 201              #: Number of grid points (must be odd)
        sys.xmax = 6.0             #: Size of the system
        sys.tmax = 1.0              #: Total real time
        sys.imax = 1000             #: Number of real time iterations
        sys.acon = 1.0              #: Smoothing of the Coloumb interaction
        sys.interaction_strength = 1#: Scales the strength of the Coulomb interaction
        sys.im = 0                  #: Use imaginary potentials
        
        def v_ext(x):
            """Initial external potential"""
            omega = 0.5
            return 0.5*(omega**2)*(x**2)
        sys.v_ext = v_ext
        
        self.pm = pm

    def test_total_energy_1(self):
        r""" Checks total energy for harmonic oscillator
        
        .. math ::  E = \sum_{k=0}^{n-1} \omega (k + \frac{1}{2})

        For n=2, omega=0.5, we find E = 0.5 * (0.5+1.5) = 1
        """
        results = NON.main(self.pm)
        self.assertAlmostEqual(results.gs_non_E, 1.0, places=3)

