"""Tests for mixing schemes
""" 
from __future__ import division
from __future__ import absolute_import

import numpy as np
import numpy.testing as nt
import unittest

from . import input
from . import mix
from . import NON

class TestPulay(unittest.TestCase):
    """Tests for the Pulay mixer
    
    """ 

    def setUp(self):
        """ Sets up harmonic oscillator system """
        pm = input.Input()
        pm.run.name = 'unittest'
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

    def test_array_update_1(self):
        """Testing internal variables of Pulay mixer
        
        Just checking that the maths works as expected from
        [Kresse1996]_ p.34 ...
        """
        pm = self.pm
        pm.lda.kerker_length = 100
        order = 4
        mixer = mix.PulayMixer(pm, order=order, preconditioner=None)

        x = np.linspace(-pm.sys.xmax, pm.sys.xmax, pm.sys.grid)
        den_in = 1 + 0.1*np.sin(x)
        den_out = 1 - 0.1*np.sin(x)

        den_in_new = mixer.mix(den_in, den_out)

        nt.assert_allclose(mixer.den_in[0], den_in)
        #nt.assert_allclose(mixer.den_delta[0], den_in-0)
        nt.assert_allclose(mixer.res[0], -0.2*np.sin(x))
        #nt.assert_allclose(mixer.res_delta[0], -0.2*np.sin(x)-0)

        #overlaps = 0.04*np.dot(np.sin(x),np.sin(x))
        #A_bar = overlaps
        #A_bar_inv = 1/overlaps
        #alpha_bar = - A_bar_inv * overlaps

        #nt.assert_allclose(alpha_bar, -1)


class TestKerker(unittest.TestCase):
    """Tests for the Kerker preconditioner
    
    """ 

    def setUp(self):
        """ Sets up harmonic oscillator system """
        pm = input.Input()
        pm.run.name = 'unittest'
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

        pm.setup_space()
        self.pm = pm


    def test_screening_length_1(self):
        """Testing screening length in Kerker
        
        Check that for infinite screening length, simple mixing is recovered.
        [Kresse1996]_ p.34 ...
        """
        pm = self.pm
        pm.lda.kerker_length = 1e6
        pm.lda.mix = 1.0

        mixer = mix.PulayMixer(pm, order=20, preconditioner='kerker')

        den = NON.main(pm).gs_non_den
        # Note: Kerker always removes G=0 cmponent
        # (it is intended to be used on density *differences*, where the G=0
        # component vanishes anyhow)
        den -= np.average(den)
        den_cond = mixer.precondition(den, None, None)


        nt.assert_allclose(den, den_cond, 1e-3)


class TestRPA(unittest.TestCase):
    """Tests for the RPA preconditioner
    
    """ 

    def setUp(self):
        """ Sets up harmonic oscillator system """
        pm = input.Input()
        pm.run.name = 'unittest'
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

        pm.lda.mix = 1.0

        self.pm = pm

    def test_chi_1(self):
        """Testing potential-density response
        
        Testing some basic symmetry properties of the
        potential-density response and the preconditioning matrices
        required for density/potential mixing.

        """
        pm = self.pm

        mixer = mix.PulayMixer(pm, order=1, preconditioner='rpa')

        results = NON.main(pm)
        den = results.gs_non_den
        eigv = results.gs_non_eigv
        eigf = results.gs_non_eigf

        chi = mixer.preconditioner.chi(eigv, eigf)
        v = mixer.preconditioner.coulomb_repulsion
        dx = mixer.preconditioner.x_delta
        nx = mixer.preconditioner.x_npt

        nt.assert_allclose(chi, chi.T, 1e-6)
        nt.assert_allclose(v, v.T, 1e-6)

        # this is just for testing purposes
        eps_pmix = np.eye(nx)/dx - np.dot(v,chi)*dx
        eps_dmix = np.eye(nx)/dx - np.dot(chi,v)*dx
        nt.assert_allclose(eps_pmix, eps_dmix.T, 1e-6)
