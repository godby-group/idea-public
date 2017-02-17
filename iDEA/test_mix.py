"""Tests for mixing schemes
""" 

import numpy as np
import numpy.testing as nt
import input
import unittest

import mix
import NON

class TestPulay(unittest.TestCase):
    """Tests for the Pulay mixer
    
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

    def test_array_update_1(self):
        """Testing internal variables of Pulay mixer
        
        Just checking that the maths works as expected from
        [Kresse1996]_ p.34 ...
        """
        pm = self.pm
        pm.lda.kerker_length = 100
        order = 4


        mixer = mix.PulayMixer(pm, order=order)
        x = np.linspace(-pm.sys.xmax, pm.sys.xmax, pm.sys.grid)
        den_in = 1 + 0.1*np.sin(x)
        den_out = 1 - 0.1*np.sin(x)

        den_in_new = mixer.mix(den_in, den_out)

        nt.assert_allclose(mixer.den_in[1], den_in)
        nt.assert_allclose(mixer.den_delta[0], den_in-0)
        nt.assert_allclose(mixer.res[1], -0.2*np.sin(x))
        nt.assert_allclose(mixer.res_delta[0], -0.2*np.sin(x)-0)

        overlaps = 0.04*np.dot(np.sin(x),np.sin(x))
        A_bar = overlaps
        A_bar_inv = 1/overlaps
        alpha_bar = - A_bar_inv * overlaps

        nt.assert_allclose(alpha_bar, -1)

        #nt.assert_allclose(den_in_new, 


class TestKerker(unittest.TestCase):
    """Tests for the Kerker preconditioner
    
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


    def test_screening_length_1(self):
        """Testing screening length in Kerker
        
        Check that for infinite screening length, simple mixing is recovered.
        [Kresse1996]_ p.34 ...
        """
        pm = self.pm
        pm.lda.kerker_length = 1e6
        pm.lda.mix = 1.0

        mixer = mix.PulayMixer(pm, order=20, preconditioner='Kerker')

        den = NON.main(pm).gs_non_den
        # Note: Kerker always removes G=0 cmponent
        # (but it is intended to be used on density *differences*)
        den -= np.average(den)
        den_cond = mixer.precondition(den, None, None)


        nt.assert_allclose(den, den_cond, 1e-3)

