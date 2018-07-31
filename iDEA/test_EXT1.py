"""Tests for 1-electron exact calculations in iDEA
""" 
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import numpy.testing as nt
import unittest

from . import EXT1
from . import input


class TestHarmonicOscillator(unittest.TestCase):
    """ Tests for the harmonic oscillator potential
    
    External potential is the harmonic oscillator (this is the default in iDEA).
    Testing ground-state and time-dependence case.                                    
    """ 

    def setUp(self):
        """ Sets up harmonic oscillator system """
        pm = input.Input()
        pm.run.name = 'unittest'
        pm.run.save = False
        pm.run.verbosity = 'low'
        pm.run.time_dependence = True

        pm.sys.NE = 1                     #: Number of electrons
        pm.sys.grid = 151                 #: Number of grid points (must be odd)
        pm.sys.stencil = 3                #: Discretisation of 2nd derivative (3 or 5 or 7)
        pm.sys.xmax = 7.5                 #: Size of the system
        pm.sys.tmax = 0.25                #: Total real time                                             
        pm.sys.imax = 251                 #: Number of real time iterations (NB: deltat = tmax/(imax-1)) 
        pm.sys.acon = 1.0                 #: Smoothing of the Coloumb interaction

        def v_ext(x):
            """Initial external potential"""
            return 0.5*(0.4**2)*(x**2)
        pm.sys.v_ext = v_ext

        def v_pert(x):
            """Time-dependent perturbation potential
            Switched on at t=0.                     
            """                                                                     
            return -0.05*x                                
        pm.sys.v_pert = v_pert                         
        
        pm.ext.rtol_solver = 1e-12        #: Tolerance of linear solver in real time propagation

        self.pm = pm

    def test_system(self):
        """Test ground-state and then real time propagation"""
        pm = self.pm
        results = EXT1.main(pm)
        den_gs = results.gs_ext_den                         
        den_td = results.td_ext_den                         
        cur = results.td_ext_cur   

        # Ground-state   
        den_analytic = np.zeros(151, dtype=np.float)               
        for j in range(151):                                  
            x = -7.5 + j*0.1                                 
            den_analytic[j] = np.sqrt(0.4/np.pi)*np.exp(-0.4*x**2)                                                    
        den_error = np.sum(np.absolute(den_gs-den_analytic))           
                       
        nt.assert_allclose(results.gs_ext_E, 0.2000, atol=1e-4) 
        nt.assert_allclose(den_error, 2.8e-3, atol=1e-4)

        # Time-dependence                               
        deltan = np.sum(np.absolute(den_td[250,:]-den_gs[:]))
        deltac = np.sum(np.absolute(cur[250,:]))             
                                                    
        nt.assert_allclose(deltan, 1.11e-2, atol=1e-4)      
        nt.assert_allclose(deltac, 1.24e-1, atol=1e-3)      


class TestAtom(unittest.TestCase):                                                         
    """ Tests for an atomic-like potential                                                   
                                                                                                         
    External potential is a softened atomic-like potential.                         
    Testing ground-state case.
    Testing 3-, 5- and 7-point stencil for the second-derivative. 
    """                                                                                                  
                                                                                                         
    def setUp(self):                                                                                     
        """ Sets up atomic system """                                                       
        pm = input.Input()      
        pm.run.name = 'unittest'                                                                         
        pm.run.save = False                                                                              
        pm.run.verbosity = 'low'                                                                        
                                                                                                
        pm.sys.NE = 1                     #: Number of electrons                                        
        pm.sys.grid = 31                  #: Number of grid points (must be odd)                                   
        pm.sys.xmax = 12.0                #: Size of the system                  
                                                                                                
        def v_ext(x):                                                                                   
            """Initial external potential"""                                                            
            return -1.0/(abs(0.1*x)+1)                                          
        pm.sys.v_ext = v_ext                                                                                 
                                                                                              
        self.pm = pm                                                                                    
                                                                              
    def test_stencil_three(self):
        """Test 3-point stencil""" 
        pm = self.pm
        pm.sys.stencil = 3                #: Discretisation of 2nd derivative (3 or 5 or 7)
        results = EXT1.main(pm)

        nt.assert_allclose(results.gs_ext_E, -0.84927, atol=1e-5)

    def test_stencil_five(self):                    
        """Test 5-point stencil"""                   
        pm = self.pm
        pm.sys.stencil = 5                #: Discretisation of 2nd derivative (3 or 5 or 7)                   
        results = EXT1.main(pm)                      
                                                 
        nt.assert_allclose(results.gs_ext_E, -0.84840, atol=1e-5)

    def test_stencil_seven(self):                    
        """Test 7-point stencil"""                   
        pm = self.pm      
        pm.sys.stencil = 7                #: Discretisation of 2nd derivative (3 or 5 or 7)       
        results = EXT1.main(pm)                      
                                                 
        nt.assert_allclose(results.gs_ext_E, -0.84834, atol=1e-5)
