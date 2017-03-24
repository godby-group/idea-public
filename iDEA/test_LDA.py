""" Tests for the local density approximation

""" 
import LDA
import input
import unittest
import numpy as np
import numpy.testing as nt

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
        
        One method uses the exchange-correlation *energy* while another
        method uses the exchange-correlation *potential*.
        """
        pm = self.pm
        results = LDA.main(pm)

        eigf = results.gs_lda_eigf
        eigv = results.gs_lda_eigv

        # check that eigenfunctions are normalised as expected
        norms = np.sum(eigf*eigf.conj(), axis=1) * pm.sys.deltax
        nt.assert_allclose(norms, np.ones(len(eigf)))

        E1 = LDA.total_energy_eigv(pm, eigv, eigf=eigf.T)
        E2 = LDA.total_energy_eigf(pm, eigf.T)

        self.assertAlmostEqual(E1,E2, delta=1e-12)

    def test_kinetic_energy_1(self):
        r"""Compares kinetic energy function vs hamiltonian
        
        One method uses the exchange-correlation *energy* while another
        method uses the exchange-correlation *potential*.
        """
        pm = self.pm
        results = LDA.main(pm)

        eigf = results.gs_lda_eigf[:pm.sys.NE].T
        #eigv = results.gs_lda_eigv

        n = LDA.electron_density(pm, eigf)
        v_ks = 0
        H = LDA.construct_hamiltonian(pm, v_ks)


        eigv = np.dot(eigf.T, np.dot(H, eigf)) * pm.sys.deltax
        T_1 = np.sum(eigv)
        T_2 = LDA.kinetic_energy(pm, eigf)

        self.assertAlmostEqual(T_1, T_2)
