"""Tests for non-interacting calculations in iDEA
""" 
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import numpy.testing as nt
import unittest

from . import NON
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

        pm.sys.NE = 2                     #: Number of electrons
        pm.sys.grid = 201                 #: Number of grid points (must be odd)
        pm.sys.stencil = 5                #: Discretisation of 2nd derivative (3 or 5 or 7)
        pm.sys.xmax = 8.0                 #: Size of the system
        pm.sys.tmax = 0.25                #: Total real time                                             
        pm.sys.imax = 251                 #: Number of real time iterations (NB: deltat = tmax/(imax-1)) 

        def v_ext(x):
            """Initial external potential"""
            return 0.5*(0.5**2)*(x**2)
        pm.sys.v_ext = v_ext

        def v_pert(x):
            """Time-dependent perturbation potential
            Switched on at t=0.                     
            """                                                                     
            return -0.05*x                                
        pm.sys.v_pert = v_pert                         
        
        pm.non.rtol_solver = 1e-12        #: Tolerance of linear solver in real time propagation

        self.pm = pm

    def test_system(self):
        """Test ground-state and then real time propagation"""
        pm = self.pm
        results = NON.main(pm)
        eigv = results.gs_non_eigv
        den_gs = results.gs_non_den                         
        den_td = results.td_non_den                         
        cur = results.td_non_cur   

        # Ground-state   
        den_analytic = np.zeros(201, dtype=np.float)               
        for j in range(201):                                  
            x = -8.0 + j*0.08                                 
            den_analytic[j] = np.sqrt(0.5/np.pi)*np.exp(-0.5*x**2)*(1.0+x**2)                                                    
        den_error = np.sum(np.absolute(den_gs-den_analytic))           
                       
        nt.assert_allclose(results.gs_non_E, 1.0000, atol=1e-4) 
        nt.assert_allclose(den_error, 2.5e-5, atol=1e-6)

        for j in range(10):
            eigv_analytic = (j+0.5)*0.5
            nt.assert_allclose(eigv[j], eigv_analytic, atol=1e-3)

        # Time-dependence                               
        deltan = np.sum(np.absolute(den_td[250,:]-den_gs[:]))
        deltac = np.sum(np.absolute(cur[250,:]))             
                                                    
        nt.assert_allclose(deltan, 2.22e-2, atol=1e-4)      
        nt.assert_allclose(deltac, 3.11e-1, atol=1e-3)      


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
                                                                                                
        pm.sys.NE = 3                     #: Number of electrons                                        
        pm.sys.grid = 101                 #: Number of grid points (must be odd)                                    
        pm.sys.xmax = 25.0                #: Size of the system                  
                                                                                                
        def v_ext(x):                                                                                   
            """Initial external potential"""                                                            
            return -1.0/(abs(0.1*x)+1)                                          
        pm.sys.v_ext = v_ext                                                                                 
                                                                                                
        self.pm = pm                                                                                    
                                                                              
    def test_stencil_three(self):
        """Test 3-point stencil""" 
        pm = self.pm
        pm.sys.stencil = 3                #: Discretisation of 2nd derivative (3 or 5 or 7)
        results = NON.main(pm)

        nt.assert_allclose(results.gs_non_E, -2.10501, atol=1e-5)

    def test_stencil_five(self):                    
        """Test 5-point stencil"""                   
        pm = self.pm
        pm.sys.stencil = 5                #: Discretisation of 2nd derivative (3 or 5 or 7)                   
        results = NON.main(pm)                      
                                                 
        nt.assert_allclose(results.gs_non_E, -2.10312, atol=1e-5)

    def test_stencil_seven(self):                    
        """Test 7-point stencil"""                   
        pm = self.pm      
        pm.sys.stencil = 7                #: Discretisation of 2nd derivative (3 or 5 or 7)       
        results = NON.main(pm)                      
                                                 
        nt.assert_allclose(results.gs_non_E, -2.10308, atol=1e-5)
