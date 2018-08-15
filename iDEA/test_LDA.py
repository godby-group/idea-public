""" Tests for the local density approximation

""" 
from __future__ import absolute_import
from . import LDA
from . import input
import unittest
import numpy as np
import numpy.testing as nt
import scipy.sparse as sps

class LDATestHarmonic(unittest.TestCase):
    """ Tests on the harmonic oscillator potential

    """ 

    def setUp(self):
        """ Sets up harmonic oscillator system """
        pm = input.Input()
        pm.run.name = 'unittest'
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

        pm.lda.NE = 2
        pm.setup_space()
        
        self.pm = pm

    def test_total_energy_1(self):
        r"""Compares total energy computed via two methods
        
        One method uses the energy eigenvalues + correction terms.
        The other uses it only for the kinetic part, while the rest
        is computed usin energy functinals.
        """
        pm = self.pm
        results = LDA.main(pm)

        eigf = results.gs_lda2_eigf
        eigv = results.gs_lda2_eigv

        # check that eigenfunctions are normalised as expected
        norms = np.sum(eigf*eigf.conj(), axis=1) * pm.sys.deltax
        nt.assert_allclose(norms, np.ones(len(eigf)))

        E1 = LDA.total_energy_eigv(pm, eigenvalues=eigv, orbitals=eigf.T)
        E2 = LDA.total_energy_eigf(pm, orbitals=eigf.T)

        self.assertAlmostEqual(E1,E2, delta=1e-12)

    def test_kinetic_energy_1(self):
        r"""Checks kinetic energy
        
        Constructs Hamiltonian with KS-potential set to zero
        and computes expectation values.
        """
        pm = self.pm
        results = LDA.main(pm)

        eigf = results.gs_lda2_eigf[:pm.sys.NE].T
        #eigv = results.gs_lda_eigv

        n = LDA.electron_density(pm, orbitals=eigf)
        v_ks = 0
        H = LDA.banded_to_full(pm, H=LDA.hamiltonian(pm, v_ks=v_ks))


        H_psi = np.dot(H,eigf)
        T_1 = 0
        for i in range(pm.sys.NE):
            T_1 += np.dot(eigf[:,i].T, H_psi[:,i]) * pm.sys.deltax
        T_2 = LDA.kinetic_energy(pm, orbitals=eigf)

        self.assertAlmostEqual(T_1, T_2)

    def test_banded_hamiltonian_1(self):
        r"""Test construction of Hamiltonian

        Hamiltonian is constructed in banded form for speed.
        This checks that the construction actually works.
        """
        pm = self.pm
        v_ext = pm.space.v_ext
        H = LDA.banded_to_full(pm, H=LDA.hamiltonian(pm, v_ks=v_ext))

        grid = pm.sys.grid
        sd = pm.space.second_derivative
        sd_ind = pm.space.second_derivative_indices
        T = -0.5 * sps.diags(sd,sd_ind,shape=(grid,grid), dtype=np.float, format='csr')
        V = sps.diags(v_ext, 0, shape=(grid,grid), dtype=np.float, format='csr')

        H2 = (T+V).toarray()

        nt.assert_allclose(H,H2)

