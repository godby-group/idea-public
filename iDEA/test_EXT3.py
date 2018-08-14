"""Tests for 3-electron exact calculations in iDEA
""" 
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import numpy.testing as nt
import unittest

from . import EXT3
from . import input


class TestHarmonicOscillator(unittest.TestCase):
    """ Tests for the harmonic oscillator potential
    
    External potential is the harmonic oscillator (this is the default in iDEA).
    Testing both ground-state non-interacting and ground-state interacting case.
    Testing time-dependent interacting case.
    """ 

    def setUp(self):                                                                                       
        """ Sets up harmonic oscillator system """                                                         
        pm = input.Input()         
        pm.run.name = 'unittest'                                                                        
        pm.run.save = False                                                                                
        pm.run.verbosity = 'low'                                                                           
                                                                                                       
        pm.sys.NE = 3                     #: Number of electrons                                           
        pm.sys.grid = 31                  #: Number of grid points (must be odd)                           
        pm.sys.stencil = 5                #: Discretisation of 2nd derivative (3 or 5 or 7)                
        pm.sys.xmax = 7.5                 #: Size of the system                                            
        pm.sys.tmax = 0.05                #: Total real time                                               
        pm.sys.imax = 51                  #: Number of real time iterations (NB: deltat = tmax/(imax-1))   
        pm.sys.acon = 1.0                 #: Smoothing of the Coloumb interaction                                       
                                                                                                       
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

        pm.ext.itol = 1e-4                #: Tolerance of imaginary time propagation
        pm.ext.itol_solver = 1e-12        #: Tolerance of linear solver in imaginary time propagation 
        pm.ext.rtol_solver = 1e-12        #: Tolerance of linear solver in real time propagation
        pm.ext.itmax = 20.0               #: Total imaginary time
        pm.ext.iimax = 1e3                #: Imaginary time iterations
        pm.ext.ideltat = pm.ext.itmax/pm.ext.iimax #: Imaginary time step (DERIVED)
        pm.ext.initial_gspsi = 'qho'      #: Initial 2 electron ground-state wavefunction                                                                                                       
                                                                    
        self.pm = pm                                                                                       
                                                                                                       
    def test_non_interacting_system_1(self):                                                               
        """Test non-interacting system"""                                                                  
        pm = self.pm                                                                                       
        pm.sys.interaction_strength = 0.0 #: Scales the strength of the Coulomb interaction                                                               
        results = EXT3.main(pm)                                                                            
                                                                                                       
        nt.assert_allclose(results.gs_ext_E, 2.250, atol=3e-3)                                            
                                                                                                       
    def test_interacting_system_1(self):                                                                   
        """Test interacting system"""                                                                      
        pm = self.pm  
        pm.sys.interaction_strength = 1.0 #: Scales the strength of the Coulomb interaction                                                                                     
        results = EXT3.main(pm)                                                                            
                                                                                                 
        nt.assert_allclose(results.gs_ext_E, 3.188, atol=2e-3)                                            
                                                                                                       
    def test_time_dependence(self):                                                                        
        """Test real time propagation"""                                                                   
        pm = self.pm                                                                                      
        pm.run.time_dependence = True     #: whether to run time-dependent calculation 
        pm.sys.interaction_strength = 1.0 #: Scales the strength of the Coulomb interaction                                                                   
        results = EXT3.main(pm)                                                                            
        den_gs = results.gs_ext_den                                                                        
        den_td = results.td_ext_den                                                                        
        cur = results.td_ext_cur                                                                           
                                                                                                       
        deltan = np.sum(np.absolute(den_td[50,:]-den_gs[:]))*pm.space.delta                                               
        deltac = np.sum(np.absolute(cur[50,:]))*pm.space.delta                                                            
                                                                                                       
        nt.assert_allclose(deltan,8.94e-5, atol=1e-7)                                                      
        nt.assert_allclose(deltac,7.51e-3, atol=1e-5)


class TestDoubleWell(unittest.TestCase):                                                         
    """ Tests for an asymmetric double-well potential                                            
                                                                                                 
    External potential is an asymmetric double-well potential (System 1 in Hodgson et al. 2016). 
    Testing ground-state interacting case.                                                       
    Testing 3-, 5- and 7-point stencil for the second-derivative.                                 
    Testing initial wavefunction by starting from the non-interacting orbitals.                  
    """                                                                                          
                                                                                                 
    def setUp(self):                                                                             
        """ Sets up double-well system """                                               
        pm = input.Input()  
        pm.run.name = 'unittest'                                                                     
        pm.run.save = False                                                                      
        pm.run.verbosity = 'low'                                                                 
                                                                                                 
        pm.sys.NE = 3                     #: Number of electrons                                 
        pm.sys.grid = 31                  #: Number of grid points (must be odd)                     
        pm.sys.xmax = 15.0                #: Size of the system                                   
        pm.sys.acon = 1.0                 #: Smoothing of the Coloumb interaction                
        pm.sys.interaction_strength = 1.0 #: Scales the strength of the Coulomb interaction      
                                                                                                 
        def v_ext(x):                                                                            
            """Initial external potential"""                                                     
            return -1.2*np.exp((-1/125)*(x-7.0)**4) - 0.9*np.exp((-1/10)*(x+6.0)**2)             
        pm.sys.v_ext = v_ext                                                                     

        pm.ext.itol = 1e-4                #: Tolerance of imaginary time propagation                 
        pm.ext.itol_solver = 1e-12        #: Tolerance of linear solver in imaginary time propagation     
        pm.ext.itmax = 50.0               #: Total imaginary time                                    
        pm.ext.iimax = 5e3                #: Imaginary time iterations
        pm.ext.ideltat = pm.ext.itmax/pm.ext.iimax #: Imaginary time step (DERIVED)                               
        pm.ext.initial_gspsi = 'non'      #: Initial 2 electron ground-state wavefunction                                                            
                                                                   
        self.pm = pm                                                                             
                                                                                                 
    def test_stencil_three(self):                                                                
        """Test 3-point stencil"""                                                             
        pm = self.pm              
        pm.sys.stencil = 3                #: Discretisation of 2nd derivative (3 or 5 or 7)                                                              
        results = EXT3.main(pm)                                                                  
                                                                                              
        nt.assert_allclose(results.gs_ext_E, -2.08, atol=3e-2)                                
                                                                                                 
    def test_stencil_five(self):                                                                 
        """Test 5-point stencil"""                                                               
        pm = self.pm      
        pm.sys.stencil = 5                #: Discretisation of 2nd derivative (3 or 5 or 7)                                                    
        results = EXT3.main(pm)                                                                  
                                                                                                 
        nt.assert_allclose(results.gs_ext_E, -2.075, atol=5e-3)                                
                                                                                                 
    def test_stencil_seven(self):                                                                
        """Test 7-point stencil"""                                                              
        pm = self.pm      
        pm.sys.stencil = 7                #: Discretisation of 2nd derivative (3 or 5 or 7)                                                         
        results = EXT3.main(pm)                                                                  
                                                                                                 
        nt.assert_allclose(results.gs_ext_E, -2.075, atol=2e-3)                                
