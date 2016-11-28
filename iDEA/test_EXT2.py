"""Tests for 2-electron exact calculations in iDEA
""" 

import numpy as np
import numpy.testing as nt
import EXT2
import input
import unittest

# decimal places for comparison of results
d = 10

class TestHarmonicOscillator(unittest.TestCase):
    """ Tests for the harmonic oscillator potential
    
    External potential is the harmonic oscillator (this is the default in iDEA).
    Testing both non-interacting and interacting case.
    """ 

    def setUp(self):
        """ Sets up harmonic oscillator system """
        pm = input.Input()
        pm.ext.ctmax = 1e4  # larger steps
        pm.ext.cdeltat = pm.ext.ctmax/(pm.ext.cimax-1)
        pm.run.verbosity = 'low'
        self.pm = pm

    def test_non_interacting_system_1(self):
        """Test interacting system"""
        pm = self.pm
        pm.sys.interaction_strength = 0.0
        results = EXT2.main(pm)

        nt.assert_allclose(results.gs_ext_E, 0.49988695, atol=1e-6)

    def test_interacting_system_1(self):
        """Test interacting system"""
        pm = self.pm
        results = EXT2.main(pm)

        nt.assert_allclose(results.gs_ext_E, 0.75310393, atol=1e-6)

