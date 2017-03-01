"""Computes ground-state charge density of a system using the Hartree-Fock approximation. The code outputs the ground-state charge density, the 
energy of the system and the Hartree-Fock orbitals. 
"""


import copy
import pickle
import numpy as np
import scipy as sp
import scipy.linalg as spla
import scipy.sparse as sps
import results as rs


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
   U = np.zeros((pm.sys.grid,pm.sys.grid))
   for i in range(pm.sys.grid):
      xi = i*pm.sys.deltax-pm.sys.xmax
      for j in range(pm.sys.grid):
         xj = j*pm.sys.deltax-pm.sys.xmax
         U[i,j] = 1.0/(abs(xi-xj) + pm.sys.acon)
   return U


def fock(pm, eigf, U):
   r"""Constructs the fock operator from a set of orbitals

    .. math:: F(x,x') = \sum_{k} \psi_{k}(x) U(x,x') \psi_{k}(x')
                       

   parameters
   ----------
   eigf : array_like
        Eigenfunction orbitals indexed as eigf[orbital_number][space_index]
   
   U : array_like
        Coulomb matrix

   returns array_like
   """
   F = np.zeros((pm.sys.grid,pm.sys.grid), dtype='complex')
   for k in range(pm.sys.NE):
      for j in range(pm.sys.grid):
         for i in range(pm.sys.grid):
            F[i,j] += -(np.conjugate(eigf[k,i])*U[i,j]*eigf[k,j])
   return F


def groundstate(pm, v_ext, v_H, F):	 	
   r"""Calculates the oribitals and ground state density for the system for a given Fock operator

    .. math:: H = K + V + F \\
              H \psi_{i} = E_{i} \psi_{i}
                       

   parameters
   ----------
   v_ext : array_like
        external potential
   v_H : array_like
        hartree potential
   F : array_like
        Fock potential

   returns array_like, array_like, array_like, array_like
        density, normalised orbitals indexed as Psi[orbital_number][space_index], energies
   """		
   
   # construct K and V
   K = -0.5*sps.diags([1, -2, 1],[-1, 0, 1], shape=(pm.sys.grid,pm.sys.grid), format='csr', dtype=complex).todense()/(pm.sys.deltax**2)     			
   V_ext = sps.diags(v_ext, 0, shape=(pm.sys.grid,pm.sys.grid), format='csr', dtype=complex).todense()
   V_H = sps.diags(v_H, 0, shape=(pm.sys.grid,pm.sys.grid), format='csr', dtype=complex).todense()
   
   # construct H
   H = np.zeros(shape=K.shape)
   H = K + V_ext + V_H
   
   # add fock matrix
   if pm.hf.fock == 1:
      H = H + F*pm.sys.deltax
      
   # solve eigen equation
   eigv, eigf = spla.eigh(H)
   eigf = eigf.T / np.sqrt(pm.sys.deltax)
   
   # calculate density
   density = np.zeros(pm.sys.grid)
   for i in range(pm.sys.NE):
      density[:] += abs(eigf[i,:])**2 
   return density, eigf, eigv


def energy(pm, density, eigf, eigv, V_H, F):	 	
   r"""Calculates the total energy of the self-consistent Hartree-Fock density                 

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
   
   E_HF = 0
   for i in range(pm.sys.NE):
      E_HF += eigv[i]
   for i in range(pm.sys.grid):
      E_HF += -0.5*(density[i]*V_H[i])*pm.sys.deltax
   for k in range(pm.sys.NE):
      for i in range(pm.sys.grid):
         for j in range(pm.sys.grid):
            E_HF += -0.5*(np.conjugate(eigf[k,i])*F[i,j]*eigf[k,j])*pm.sys.deltax*pm.sys.deltax
   return E_HF.real
   
   
def main(parameters):
   r"""Performs Hartree-fock calculation

   parameters
   ----------
   parameters : object
      Parameters object

   returns object
      Results object
   """
   pm = parameters
   
   # Construct external potential and initial orbitals
   V_H = np.zeros(pm.sys.grid)
   F = np.zeros((pm.sys.grid,pm.sys.grid), dtype='complex')
   V_ext = np.zeros(pm.sys.grid)
   for i in range(pm.sys.grid):
      x = i * pm.sys.deltax - pm.sys.xmax
      V_ext[i] = pm.sys.v_ext(x)
   density, eigf, eigv = groundstate(pm, V_ext, V_H, F)
   
   # Construct coulomb matrix
   U = coulomb(pm)
   
   # Calculate ground state density
   con = 1.0
   while con > pm.hf.con:
      density_old = copy.deepcopy(density)
      
      # Calculate new potentials form new orbitals
      V_H_new = hartree(pm, U, density)
      F_new = fock(pm, eigf, U)
    
      # Stability mixing
      V_H = (1-pm.hf.nu)*V_H + pm.hf.nu*V_H_new 
      F = (1-pm.hf.nu)*F + pm.hf.nu*F_new # Note: this is required!
      
      # Solve KS equations
      density, eigf, eigv = groundstate(pm, V_ext, V_H, F)
      
      con = sum(abs(density-density_old))
      string = 'HF: computing ground-state density, convergence = ' + str(con)
      pm.sprint(string, 1, newline=False)
   print
   
   # Calculate ground state energy
   E_HF = energy(pm, density, eigf, eigv, V_H, F)
   pm.sprint('HF: hartree-fock energy = {}'.format(E_HF.real), 1, newline=True)
   
   results = rs.Results()
   results.add(E_HF,'gs_hf_E')
   results.add(density,'gs_hf_den')

   if pm.hf.save_eig:
       results.add(eigf, 'gs_hf_eigf')
       results.add(eigv, 'gs_hf_eigv')

   if pm.run.save:
      results.save(pm)
 
   return results

