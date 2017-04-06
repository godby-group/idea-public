"""Tests for direct minimizers
""" 

import numpy as np
import numpy.testing as nt
import input
import unittest

import minimize
import scipy.sparse as sps
import scipy.linalg as spla
import LDA

machine_epsilon = np.finfo(float).eps

class TestCG(unittest.TestCase):
    """Tests for the conjugate gradient minimizer
    
    """ 

    def setUp(self):
        """ Sets up harmonic oscillator system """
        pm = input.Input()
        pm.run.save = False
        pm.run.verbosity = 'low'

        self.pm = pm

    def test_steepest_dirs(self):
        """Testing orthogonalisation in steepest descent
        
        Just checking that the efficient numpy routines do the same
        as more straightforward loop-based techniques
        """
        pm = self.pm
        minimizer = minimize.CGMinimizer(pm, total_energy='dummy')

        # prepare Hamiltonian
        sys = pm.sys
        T = -0.5*sps.diags([1, -2, 1],[-1, 0, 1], shape=(sys.grid,sys.grid), format='csr')/(sys.deltax**2)
        x = np.linspace(-sys.xmax,sys.xmax,sys.grid)
        V = sps.diags(sys.v_ext(x), 0, shape=(sys.grid, sys.grid), dtype=np.float, format='csr')
        H = (T+V).toarray()
        energies, wfs = spla.eigh(H)

        steepest_orth_all = minimizer.steepest_dirs(wfs, H)

        # repeat orthogonalization for one single wave function
        i = 2 # could be any other state as well
        wf = wfs[:,i]
        steepest = -(np.dot(H,wf) - energies[i] * wf)

        overlaps = np.dot(wfs.conj(), steepest)
        steepest_orth = steepest
        for j in range(len(wfs)):
            if j != i: 
                steepest_orth -= overlaps[j] * wfs[j]

        nt.assert_almost_equal(steepest_orth_all[i], steepest_orth)

        # masked variant
        overlaps = np.ma.array(np.dot(wfs.conj().T, steepest), mask=False)
        overlaps.mask[i] = True
        steepest_orth = steepest - np.ma.dot(overlaps.T, wfs)
        steepest_orth = np.ma.getdata(steepest_orth)

        nt.assert_almost_equal(steepest_orth_all[i], steepest_orth)

        # unmasked variant
        overlaps = np.dot(wfs.conj().T, steepest)
        steepest_orth = steepest - np.dot(overlaps.T, wfs)

        nt.assert_almost_equal(steepest_orth_all[i], steepest_orth)

    def test_conjugate(self):
        r"""Check that gradient is computed correctly
        
        Use conjugate-gradient method on a quadratic function in
        n-dimensional space, where it is guaranteed to converge
        in n steps.

        Task: minimize the function
        
        .. math ::
            E(v) = \sum_{i=1}^2 v_i^* H v_i
            \triangle_{v_i} E = H v_i
            
        for a fixed matrix H and a set of orthonormal vectors v_i.
        
        Note: The dimension n of the vector space is the grid spacing
        times number of electrons.
        """
        pm = self.pm
        pm.sys.grid = 10  
        sqdx = np.sqrt(pm.sys.deltax)
        pm.setup()

        ndim = pm.sys.grid * pm.sys.NE

        E = lambda wfs: (wfs.conj() * np.dot(H, wfs)).sum()
        E_scaled = lambda pm, wfs: E(wfs*sqdx)

        # do not restart the cg
        minimizer = minimize.CGMinimizer(pm, 
                total_energy=E_scaled, 
                cg_restart=ndim+1,
                line_fit='linear')

        # start with random guess
        np.random.seed(1)
        wfs = np.random.rand(pm.sys.grid,pm.sys.NE)
        wfs = minimize.orthonormalize(wfs)
        H = LDA.banded_to_full(LDA.hamiltonian(pm, wfs=wfs/sqdx))

        # keeping H constant, we should find the minimum in ndim steps
        # (if it weren't for the orthonormality condition!...)
        for i in range(ndim):
            wfs = minimizer.step(wfs/sqdx,H) * sqdx
        E_1 = E(wfs)

        # compute lowest energy using diagonalisation
        energies_2, wfs_2 = spla.eigh(H)
        E_2 = np.sum(energies_2[:pm.sys.NE])

        nt.assert_allclose(E_1, E_2)


    def test_orthonormalisation(self):
        """Testing orthonormalisation of a set of vectors

        minimize.
        This should do the same as Gram-Schmidt
        """

        m = 8   # dimension of space
        n = 4   # number of vectors
        vecs = np.random.rand(m,n)

        q = np.empty((m,n))
        # the regular Gram-Schmidt algorithm
        for i in range(n):
            v = vecs[:,i]
            for j in range(i):
                v -= np.dot(v,q[:,j]) / np.dot(q[:,j],q[:,j]) * q[:,j]
            v /= np.linalg.norm(v)
            q[:,i] = v

        # is q an orthogonal matrix?
        nt.assert_almost_equal(np.dot(q.T,q), np.identity(n))
        
        q2 = minimize.orthonormalize(vecs)
        # is q2 an orthogonal matrix?
        nt.assert_almost_equal(np.dot(q2.T, q2), np.identity(n))

        # are q and q2 the same?
        nt.assert_almost_equal(q, q2)

class TestCGLDA(unittest.TestCase):
    """Tests for the conjugate gradient minimizer

    On an actual LDA system
    
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
        
        pm.lda.save_eig = True
        pm.lda.scf_type = 'cg'

        pm.setup()
        self.pm = pm

    def test_energy_derivative(self):
        r"""Compare analytical total energy derivative vs finite differences

        .. math::
            \frac{dE}{d\lambda}(\lambda) \approx \frac{E(\lambda)+E(\lambda+\delta)}{\delta} 
        """
        pm = self.pm
        pm.lda.max_iter = 5  # just start, don't solve it perfectly

        results = LDA.main(pm)  # solving using linear mixing
        vks = results.gs_lda_vks
        n = results.gs_lda_den
        H = LDA.banded_to_full(LDA.hamiltonian(pm, vks))
        wfs = results.gs_lda_eigf[:pm.sys.NE].T * np.sqrt(pm.sys.deltax)

        # compute current conjugate direction
        minimizer = minimize.CGMinimizer(pm, total_energy=LDA.total_energy_eigf)
        steepest = minimizer.steepest_dirs(wfs, H)
        conjugate = minimizer.conjugate_directions(steepest, wfs)
        dE_dL = 2 * np.sum(minimizer.braket(conjugate, H, wfs))

        # compute finite difference derivative
        delta = 1.0e-4
        E_1 = minimizer.total_energy(wfs)
        wfs_new = minimize.orthonormalize(wfs + delta * conjugate)
        E_2 = minimizer.total_energy(wfs_new)
        dE_dL_finite = (E_2 - E_1) / delta
        wfs_new = minimize.orthonormalize(wfs + delta * conjugate)
        
        # this can be quite close to machine epsilon
        # for the energy difference
        atol = np.abs(E_1) * 10 * machine_epsilon / delta
        rtol = atol / np.abs(dE_dL_finite)
        nt.assert_allclose(dE_dL, dE_dL_finite, rtol=rtol)
