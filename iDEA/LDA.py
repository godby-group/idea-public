"""Computes approximations to VKS, VH, VXC using the LDA self consistently. For ground state calculations the code outputs the LDA orbitals 
and energies of the system, the ground-state charge density and the Kohn-Sham potential. For time dependent calculation the code also outputs 
the time-dependent charge and current densities anjd the time-dependent Kohn-Sham potential. Note: Uses the [adiabatic] local density 
approximations ([A]LDA) to calculate the [time-dependent] electron density [and current] for a system of N electrons.
"""


import pickle
import numpy as np
import scipy as sp
import copy as copy
import RE_Utilities
import scipy.sparse as sps
import scipy.linalg as spla
import scipy.sparse.linalg as spsla
import results as rs


def groundstate(pm, v_KS):
   r"""Calculates the oribitals and ground state density for the system for a given potential

    .. math:: H \psi_{i} = E_{i} \psi_{i}
                       
   parameters
   ----------
   v_KS : array_like
        KS potential

   returns array_like, array_like, array_like
        density, normalised orbitals indexed as Psi[orbital_number][space_index], energies
   """	
   
   T = np.zeros((2,pm.sys.grid),dtype='float') # Kinetic Energy operator
   T[0,:] = np.ones(pm.sys.grid)/pm.sys.deltax**2 # Define kinetic energy operator							
   T[1,:] = -0.5*np.ones(pm.sys.grid)/pm.sys.deltax**2 
   H = copy.copy(T) # kinetic energy
   H[0,:] += v_KS[:]
   e,eig_func = spla.eig_banded(H,True) 
   n = np.zeros(pm.sys.grid,dtype='float')
   for i in range(pm.sys.NE):
      n[:] += abs(eig_func[:,i])**2 # Calculate density
   n[:] /= pm.sys.deltax # Normalise
   eig_func = eig_func / np.sqrt(pm.sys.deltax) 
   return n,eig_func,e



def hartree(pm, U, density):
   r"""Constructs the hartree potential for a given density

   .. math::

       V_{H} \left( x \right) = \int_{\forall} U\left( x,x' \right) n \left( x'\right) dx'

   parameters
   ----------
   U : array_like
        Coulomb matrix
        
   density : array_like
        given density

   returns array_like
   """
   return np.dot(U,density)*pm.sys.deltax


def coulomb(pm):
   r"""Constructs the coulomb matrix

   .. math::

       U \left( x,x' \right) = \frac{1}{|x-x'| + 1}

   parameters
   ----------

   returns array_like
   """
   U = np.zeros((pm.sys.grid,pm.sys.grid),dtype='float')
   for i in xrange(pm.sys.grid):
      for k in xrange(pm.sys.grid):	
         U[i,k] = 1.0/(abs(i*pm.sys.deltax-k*pm.sys.deltax)+pm.sys.acon)
   return U


def XC(pm , Den): 
   r"""Finite LDA approximation for the Exchange-Correlation potential

   parameters
   ----------
   Den : array_like
        density

   returns array_like
        Exchange-Correlation potential
   """
   V_xc = np.zeros(pm.sys.grid,dtype='float')
   if (pm.sys.NE == 1):
      V_xc[:] = ((-1.315+2.16*Den[:]-1.71*(Den[:])**2)*Den[:]**0.638) 
   elif (pm.sys.NE == 2):
      V_xc[:] = ((-1.19+1.77*Den[:]-1.37*(Den[:])**2)*Den[:]**0.604) 
   else:
      V_xc[:] = ((-1.24+2.1*Den[:]-1.7*(Den[:])**2)*Den[:]**0.61) 
   return V_xc


def EXC(pm, Den): 
   r"""Finite LDA approximation for the Exchange-Correlation energy

   parameters
   ----------
   Den : array_like
        density

   returns float
        Exchange-Correlation energy
   """
   E_xc_LDA = 0.0
   if (pm.sys.NE == 1):
      for i in xrange(pm.sys.grid):
         e_xc_LDA = ((-0.803+0.82*Den[i]-0.47*(Den[i])**2)*Den[i]**0.638) 
         E_xc_LDA += (Den[i])*(e_xc_LDA)*pm.sys.deltax
   elif (pm.sys.NE == 2):
      for i in xrange(pm.sys.grid):
         e_xc_LDA = ((-0.74+0.68*Den[i]-0.38*(Den[i])**2)*Den[i]**0.604) 
         E_xc_LDA += (Den[i])*(e_xc_LDA)*pm.sys.deltax
   else:
      for i in xrange(pm.sys.grid):
         e_xc_LDA = ((-0.77+0.79*Den[i]-0.48*(Den[i])**2)*Den[i]**0.61)
         E_xc_LDA += (Den[i])*(e_xc_LDA)*pm.sys.deltax
   return E_xc_LDA


def energy(pm, density, eigf, eigv, V_H, V_xc):	 	
   r"""Calculates the total energy of the self-consistent LDA density                 

   parameters
   ----------
   pm : array_like
        external potential
   density : array_like
		  density
   eigf : array_like
        eigenfunctions
   eigv : array_like
        eigenvalues
   V_H : array_like
        Hartree potential
   F : array_like
        Fock potential

   returns float
   """		
   
   E_LDA = 0.0
   for i in range(pm.sys.NE):
      E_LDA += eigv[i]
   for i in range(pm.sys.grid):
      E_LDA += -0.5*(density[i]*V_H[i])*pm.sys.deltax
   for i in range(pm.sys.grid):
      E_LDA += -1.0*(density[i]*V_xc[i])*pm.sys.deltax
   E_LDA += EXC(pm, density)
   return E_LDA.real
   
   
def CalculateCurrentDensity(pm, n, j):
   r"""Calculates the current density of a time evolving wavefunction by solving the continuity equation.

   .. math::

       \frac{\partial n}{\partial t} + \nabla \cdot j = 0
       
   Note: This function requires RE_Utilities.so

   parameters
   ----------
   total_td_density : array_like
      Time dependent density of the system indexed as total_td_density[time_index][space_index]

   returns array_like
      Time dependent current density indexed as current_density[time_index][space_index]
   """
   J = RE_Utilities.continuity_eqn(pm.sys.grid,pm.sys.deltax,pm.sys.deltat,n[j,:],n[j-1,:])
   if pm.sys.im == 1:
      for j in xrange(pm.sys.grid):
         for k in xrange(j+1):
            x = k*pm.sys.deltax-pm.sys.xmax
            J[j] -= abs(pm.sys.v_pert_im(x))*n[j,k]*pm.sys.deltax
   return J


def CrankNicolson(pm, v, Psi, n, j): 
   r"""Solves Crank Nicolson Equation
   """
   Mat = LHS(pm, v,j)
   Mat = Mat.tocsr()
   Matin =- (Mat-sps.identity(pm.sys.grid,dtype='complex'))+sps.identity(pm.sys.grid,dtype='complex')
   for i in range(pm.sys.NE):
      B = Matin*Psi[i,j-1,:]
      Psi[i,j,:] = spsla.spsolve(Mat,B)
      n[j,:] = 0
      for i in range(pm.sys.NE):
         n[j,:] += abs(Psi[i,j,:])**2
   return n,Psi


def LHS(pm, v, j):	
   r"""Constructs the matrix A to be used in the crank-nicholson solution of Ax=b when evolving the wavefunction in time (Ax=b)
   """
   CNLHS = sps.lil_matrix((pm.sys.grid,pm.sys.grid),dtype='complex') # Matrix for the left hand side of the Crank Nicholson method										
   for i in xrange(pm.sys.grid):
      CNLHS[i,i] = 1.0+0.5j*pm.sys.deltat*(1.0/pm.sys.deltax**2+v[j,i])
      if i < pm.sys.grid-1:
         CNLHS[i,i+1] = -0.5j*pm.sys.deltat*(0.5/pm.sys.deltax**2)
      if i > 0:
         CNLHS[i,i-1] = -0.5j*pm.sys.deltat*(0.5/pm.sys.deltax**2)
   return CNLHS

# Main function
def main(parameters):
   r"""Performs LDA calculation

   parameters
   ----------
   parameters : object
      Parameters object

   returns object
      Results object
   """
   pm = parameters
   
   v_s = np.zeros(pm.sys.grid,dtype='float')
   v_ext = np.zeros(pm.sys.grid,dtype='float')
   Psi = np.zeros((pm.sys.NE,pm.sys.imax,pm.sys.grid), dtype='complex')
   for i in xrange(pm.sys.grid):
      v_s[i] = pm.sys.v_ext((i*pm.sys.deltax-pm.sys.xmax)) # External potential
      v_ext[i] = pm.sys.v_ext((i*pm.sys.deltax-pm.sys.xmax)) # External potential
   n,waves,energies = groundstate(pm, v_s) #Inital guess
   U = coulomb(pm) # Create Coulomb matrix
   n_old = np.zeros(pm.sys.grid,dtype='float')
   n_old[:] = n[:] 
   convergence = 1.0
   iteration = 1
   while convergence > pm.lda.tol and iteration < pm.lda.max_iter: # Use LDA
      v_s_old = copy.copy(v_s)
      if pm.lda.mix == 0:
         v_s[:] = v_ext[:]+hartree(pm, n, U)+XC(pm, n)
      else:
         v_s[:] = (1-pm.lda.mix)*v_s_old[:]+pm.lda.mix*(v_ext[:]+hartree(pm, n, U)+XC(pm, n))
      n,waves,energies = groundstate(pm, v_s) # Calculate LDA density 
      convergence = np.sum(abs(n-n_old))*pm.sys.deltax
      n_old[:] = n[:]
      string = 'LDA: electron density convergence = ' + str(convergence)
      pm.sprint(string,1,newline=False)
      iteration += 1
   if iteration > pm.lda.max_iter:
      pm.sprint('LDA: Warning: Reached maximum number of iterations. Terminating',1)

   pm.sprint('',1)
   

   v_h = hartree(pm, n,U)
   v_xc = XC(pm, n)
   LDA_E = energy(pm, n, waves, energies, v_h, v_xc)
   pm.sprint('LDA: ground-state energy: {}'.format(LDA_E),1)
   
   results = rs.Results()
   results.add(v_s[:], 'gs_lda_vks')
   results.add(v_h[:], 'gs_lda_vh')
   results.add(v_xc[:], 'gs_lda_vxc')
   results.add(n[:], 'gs_lda_den')
   results.add(LDA_E, 'gs_lda_E')

   if pm.lda.save_eig:
       results.add(waves.T,'gs_lda_eigf')
       results.add(energies,'gs_lda_eigv')

   if pm.run.save:
      results.save(pm)

   if pm.run.time_dependence == True:
      for i in range(pm.sys.NE):
         Psi[i,0,:] = waves[:,i]
      v_s_t = np.zeros((pm.sys.imax,pm.sys.grid),dtype='float')
      v_xc_t = np.zeros((pm.sys.imax,pm.sys.grid),dtype='float')
      current = np.zeros((pm.sys.imax,pm.sys.grid),dtype='float')
      n_t = np.zeros((pm.sys.imax,pm.sys.grid),dtype='float')
      v_s_t[0,:] = v_s[:]
      n_t[0,:] = n[:]
      for i in xrange(pm.sys.grid): 
         v_s_t[1,i] = v_s[i]+pm.sys.v_pert((i*pm.sys.deltax-pm.sys.xmax))  
         v_ext[i] += pm.sys.v_pert((i*pm.sys.deltax-pm.sys.xmax)) 
      for j in range(1,pm.sys.imax): 
         string = 'LDA: evolving through real time: t = ' + str(j*pm.sys.deltat) 
         pm.sprint(string,1,newline=False)
         n_t,Psi = CrankNicolson(pm, v_s_t,Psi,n_t,j)
         if j != pm.sys.imax-1:
            v_s_t[j+1,:] = v_ext[:]+hartree(pm, n_t[j,:],U)+XC(pm, n_t[j,:])
         current[j,:] = CalculateCurrentDensity(pm, n_t,j)
         v_xc_t[j,:] = XC(pm, n_t[j,:])

      # Output results
      results.add(v_s_t, 'td_lda_vks')
      results.add(v_xc_t, 'td_lda_vxc')
      results.add(n_t, 'td_lda_den')
      results.add(current, 'td_lda_cur')

      if pm.run.save:
         l = ['td_lda_vks','td_lda_vxc','td_lda_den','td_lda_cur']
         results.save(pm, list=l)

      pm.sprint('',1)
   return results
