""" Tests for reverse-engineering

"""
from __future__ import absolute_import
from . import RE
from . import input
import numpy as np
import numpy.testing as nt
import unittest

# decimal places for comparison of results
d = 6

class RETest(unittest.TestCase):
    """ Tests on the harmonic oscillator potential
    """

    def setUp(self):
        """ Sets up harmonic oscillator system """
        pm = input.Input()
        pm.run.name = 'unittest'
        pm.run.save = True
        pm.run.verbosity = 'low'

        ### system parameters
        sys = pm.sys
        sys.NE = 2                  #: Number of electrons
        sys.grid = 61               #: Number of grid points (must be odd)
        sys.xmax = 6.0              #: Size of the system
        sys.tmax = 1.0              #: Total real time
        sys.imax = 1000             #: Number of real time iterations
        sys.acon = 1.0              #: Smoothing of the Coloumb interaction
        sys.interaction_strength = 1#: Scales the strength of the Coulomb interaction
        sys.im = 0                  #: Use imaginary potentials

        self.pm = pm

    def test_total_energy_1(self):
        r""" Checks RE for non-interacting system (harmonic oscillator)

        Starts from the LDA Vks, should give Vks = Vxt.
        """
        pm = self.pm
        pm.run.NON = True
        pm.run.LDA = True
        pm.lda.NE = 2
        pm.re.gs_density_tolerance = 1e-6
        pm.re.starting_guess = 'lda2'
        results = pm.execute()
        rev = RE.main(pm, 'non')
        nt.assert_array_almost_equal(results.non.gs_non_den, rev.gs_nonre_den, decimal=d)
