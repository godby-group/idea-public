"""Tests for direct minimizers
""" 

import numpy as np
import numpy.testing as nt
import input
import unittest

import minimize
import NON
import scipy.sparse as sps
import scipy.linalg as spla

class TestCG(unittest.TestCase):
    """Tests for the conjugate gradient minimizer
    
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
        
        pm.ext.ctol = 1e-5

        self.pm = pm

    def test_steepest_dirs(self):
        """Testing orthogonalisation in steepest descent
        
        Just checking that the efficient numpy routines do the same
        as more straightforward loop-based techniques
        """
        pm = self.pm
        minimizer = minimize.CGMinimizer(pm)

        # prepare Hamiltonian
        sys = pm.sys
        T = -0.5*sps.diags([1, -2, 1],[-1, 0, 1], shape=(sys.grid,sys.grid), format='csr')/(sys.deltax**2)
        x = np.linspace(-sys.xmax,sys.xmax,sys.grid)
        V = sps.diags(sys.v_ext(x), 0, shape=(sys.grid, sys.grid), dtype=np.float, format='csr')
        H = (T+V).toarray()
        energies, wfs = spla.eigh(H)
        wfs = wfs.T

        steepest_orth_all = minimizer.steepest_dirs(H, wfs, energies)

        # repeat orthogonalization for one single wave function
        i = 2 # could be any other state as well
        wf = wfs[i]
        steepest = -np.dot(H,wf) - energies[i] * wf

        overlaps = np.dot(wfs.conj(), steepest)
        steepest_orth = steepest
        for j in range(len(wfs)):
            if j != i: 
                steepest_orth -= overlaps[j] * wfs[j]

        # masked variant
        #overlaps = np.ma.array(np.dot(wfs.conj(), steepest), mask=False)
        #overlaps.mask[i] = True
        #steepest_orth = steepest - np.ma.dot(overlaps.T, wfs)
        #steepest_orth = np.ma.getdata(steepest_orth)

        nt.assert_almost_equal(steepest_orth_all[i], steepest_orth)

