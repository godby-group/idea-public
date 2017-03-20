""" Tests for the local density approximation

""" 
import LDA
import input
import unittest
import numpy as np
import numpy.testing as nt

# decimal places for comparison of results
d = 6

class LDATestHarmonic(unittest.TestCase):
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

        pm.lda.save_eig = True
        
        pm.setup()
        self.pm = pm

    def test_total_energy_1(self):
        r"""Compares total energy computed via two methods
        
        """
        pm = self.pm
        results = LDA.main(pm)

        eigf = results.gs_lda_eigf
        # check eigenfunctions are normalised as expected
        norms = np.sum(eigf*eigf.conj(), axis=1) * pm.sys.deltax
        nt.assert_allclose(norms, np.ones(len(eigf)))
        eigv = results.gs_lda_eigv

        E1 = LDA.total_energy(pm, eigv, eigf=eigf.T)
        E2 = LDA.total_energy_2(pm, eigf.T)

        self.assertAlmostEqual(E1,E2)

