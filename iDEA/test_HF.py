""" Tests for the local density approximation

"""
from __future__ import absolute_import
from . import LDA
from . import input
import unittest
import numpy as np
import numpy.testing as nt
import scipy.sparse as sps

class HFTestHarmonic(unittest.TestCase):
    """ Tests on the harmonic oscillator potential
    """

    def setUp(self):
        """ Sets up harmonic oscillator system """
        pm = input.Input()
        pm.run.name = 'unittest'
        pm.run.HF = True
        pm.run.EXT = True
        pm.run.save = False
        pm.run.verbosity = 'low'

        ### system parameters
        sys = pm.sys
        sys.NE = 1                   #: Number of electrons
        sys.grid = 201               #: Number of grid points (must be odd)
        sys.xmax = 6.0               #: Size of the system
        sys.tmax = 1.0               #: Total real time
        sys.imax = 1000              #: Number of real time iterations
        sys.acon = 1.0               #: Smoothing of the Coloumb interaction
        sys.interaction_strength = 1 #: Scales the strength of the Coulomb interaction
        sys.im = 0                   #: Use imaginary potentials

        def v_ext(x):
            """Initial external potential"""
            omega = 0.5
            return 0.5*(omega**2)*(x**2)
        sys.v_ext = v_ext

        pm.setup_space()
        self.pm = pm

    def test_1_electron(self):
        r"""Ensures HF is exact for one electron
        """
        pm = self.pm
        results = pm.execute()

        den_ext = results.ext.gs_ext_den
        den_hf = results.hf.gs_hf_den

        # check that HF and exact density are identical for 1 electron
        nt.assert_allclose(den_ext, den_hf)
