"""Tests for 3-electron exact calculations in iDEA
""" 
from __future__ import absolute_import

import numpy as np
import numpy.testing as nt
from . import EXT3
from . import input
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
        pm.sys.NE = 3                  #: Number of electrons
        pm.sys.grid = 41               #: Number of grid points (must be odd)
        pm.sys.stencil = 7             #: Stencil 
        pm.sys.xmax = 7.5              #: Size of the system
        pm.sys.acon = 1.0              #: Smoothing of the Coloumb interaction
        pm.sys.interaction_strength = 1#: Scales the strength of the Coulomb interaction
        def v_ext(x):
            """Initial external potential"""
            return 0.5*(0.5**2)*(x**2)
        pm.sys.v_ext = v_ext
        
        pm.ext.itol = 1e-5

        self.pm = pm

    def test_non_interacting_system_1(self):
        """Test non-interacting system"""
        pm = self.pm
        pm.sys.interaction_strength = 0.0
        results = EXT3.main(pm)

        nt.assert_allclose(results.gs_ext_E, 2.250, atol=1e-3)

    def test_interacting_system_1(self):
        """Test interacting system"""
        pm = self.pm
        pm.sys.stencil = 5
        pm.sys.grid = 31
        results = EXT3.main(pm)

        nt.assert_allclose(results.gs_ext_E, 3.186, atol=1e-3)

