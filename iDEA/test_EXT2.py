"""Tests for 2-electron exact calculations in iDEA
""" 

import numpy as np
import numpy.testing as nt
import EXT2
import input
import unittest

class TestHarmonicOscillator(unittest.TestCase):
    """ Tests for the harmonic oscillator potential
    
    External potential is the harmonic oscillator (this is the default in iDEA).
    Testing both non-interacting and interacting case.
    """ 

    def setUp(self):
        """ Sets up harmonic oscillator system """
        pm = input.Input()
        pm.run.save = False
        pm.run.verbosity = 'low'

        # It might still be possible to speed this up
        pm.sys.NE = 2                  #: Number of electrons
        pm.sys.grid = 61               #: Number of grid points (must be odd)
        pm.sys.xmax = 7.5              #: Size of the system
        pm.sys.acon = 1.0              #: Smoothing of the Coloumb interaction
        pm.sys.interaction_strength = 1#: Scales the strength of the Coulomb interaction
        def v_ext(x):
            """Initial external potential"""
            return 0.5*(0.25**2)*(x**2)
        pm.sys.v_ext = v_ext
        
        pm.ext.itol = 1e-5

        self.pm = pm

    def test_non_interacting_system_1(self):
        """Test interacting system"""
        pm = self.pm
        pm.sys.interaction_strength = 0.0
        results = EXT2.main(pm)

        #nt.assert_allclose(results.gs_ext_E, 0.49988695, atol=1e-6)
        nt.assert_allclose(results.gs_ext_E, 0.500, atol=1e-3)

    def test_interacting_system_1(self):
        """Test interacting system"""
        pm = self.pm
        results = EXT2.main(pm)

        nt.assert_allclose(results.gs_ext_E, 0.753, atol=1e-3)

