"""Computes ground-state charge density in the Hartree-Fock approximation. 

The code outputs the ground-state charge density, the energy of the system and
the Hartree-Fock orbitals. 
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import


import copy
import pickle
import numpy as np
import scipy as sp
import scipy.linalg as spla
import scipy.sparse as sps
from . import results as rs


def hartree(pm, density):
   r"""Computes Hartree potential for a given density

   .. math::

       V_H(r) = = \int U(r,r') n(r')dr'

   parameters
   ----------
   density : array_like
        given density

   returns array_like
   """
   return np.dot(pm.space.v_int,density)*pm.sys.deltax


def fock(pm, eigf):
   r"""Constructs Fock operator from a set of orbitals

    .. math:: F(x,x') = \sum_{k} \psi_{k}(x) U(x,x') \psi_{k}(x')
 
    where U(x,x') denotes the appropriate Coulomb interaction.
                       
   parameters
   ----------
   eigf : array_like
        Eigenfunction orbitals indexed as eigf[space_index][orbital_number]

   returns
   -------
   F: array_like
     Fock matrix
   """
   F = np.zeros((pm.sys.grid,pm.sys.grid), dtype='complex')
   #for k in range(pm.sys.NE):
   #   for j in range(pm.sys.grid):
   #      for i in range(pm.sys.grid):
   #         F[i,j] += -(np.conjugate(eigf[k,i])*U[i,j]*eigf[k,j])

   for i in range(pm.sys.NE):
       orb = eigf[:,i]
       F -= np.tensordot(orb.conj(), orb, axes=0)
   F = F * pm.space.v_int

   return F


def electron_density(pm, orbitals):
    r"""Compute density for given orbitals

    parameters
    ----------
    orbitals: array_like
      array of properly normalised orbitals[space-index,orital number]

    returns
    -------
    n: array_like
      electron density
    """
    occupied = orbitals[:, :pm.sys.NE]
    n = np.sum(occupied*occupied.conj(), axis=1)
    return n


def hamiltonian(pm, wfs):
    r"""Compute HF Hamiltonian

    Computes HF Hamiltonian from a given set of single-particle states

    parameters
    ----------
    wfs : array_like
         single-particle states

    returns array_like
         Hamiltonian matrix
    """
    sd = pm.space.second_derivative
    sd_ind = pm.space.second_derivative_indices

    # construct kinetic energy
    K = -0.5*sps.diags(sd, sd_ind, shape=(pm.sys.grid,pm.sys.grid), format='csr', dtype=complex)

    # construct external and hartree potential
    n = electron_density(pm, wfs)
    V = sps.diags(pm.space.v_ext + hartree(pm,n), 0, shape=(pm.sys.grid,pm.sys.grid), format='csr', dtype=complex)
    
    # construct H
    H = (K+V).toarray()
    
    # add fock matrix
    if pm.hf.fock == 1:
       H = H + fock(pm,wfs) * pm.sys.deltax

    return H

def groundstate(pm, H):
   r"""Diagonalises Hamiltonian H

    .. math:: H = K + V + F \\
              H \psi_{i} = E_{i} \psi_{i}
                       
   parameters
   ----------
   H: array_like
     Hamiltonian matrix (band form)

   returns
   -------
   n: array_like
     density
   eigf: array_like
     normalised orbitals, index as eigf[space_index,orbital_number]
   eigv: array_like
     orbital energies

   """		
      
   # solve eigen equation
   eigv, eigf = spla.eigh(H)
   eigf = eigf/ np.sqrt(pm.sys.deltax)
   
   # calculate density
   n = electron_density(pm,eigf)

   return n, eigf, eigv


def total_energy(pm, eigf, eigv):
   r"""Calculates total energy of Hartree-Fock wave function

   parameters
   ----------
   pm : array_like
        external potential
   eigf : array_like
        eigenfunctions
   eigv : array_like
        eigenvalues

   returns float
   """		
   

   E_HF = 0
   E_HF += np.sum(eigv[:pm.sys.NE])

   # Subtract Hartree energy
   n = electron_density(pm, eigf)
   V_H = hartree(pm, n)
   E_HF -= 0.5 * np.dot(V_H,n) * pm.sys.deltax

   # Fock correction
   F = fock(pm,eigf)
   for k in range(pm.sys.NE):
      orb = eigf[:,k]
      E_HF -= 0.5 * np.dot(orb.conj().T, np.dot(F, orb)) * pm.sys.deltax**2
   return E_HF.real
   

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
      for j in range(pm.sys.grid):
         for k in range(j+1):
            x = k*pm.sys.deltax-pm.sys.xmax
            J[j] -= abs(pm.sys.v_pert_im(x))*n[j,k]*pm.sys.deltax
   return J


def LHS(pm, v, j):
   r"""Constructs the matrix A to be used in the crank-nicholson solution of Ax=b when evolving the wavefunction in time (Ax=b)

   .. math::
       A = \mathbb{1} + \frac{1}{2}H i dt

   """
   CNLHS = sps.lil_matrix((pm.sys.grid,pm.sys.grid),dtype='complex') # Matrix for the left hand side of the Crank Nicholson method
   for i in range(pm.sys.grid):
      CNLHS[i,i] = 1.0+0.5j*pm.sys.deltat*(1.0/pm.sys.deltax**2+v[j,i])
      if i < pm.sys.grid-1:
         CNLHS[i,i+1] = -0.5j*pm.sys.deltat*(0.5/pm.sys.deltax**2)
      if i > 0:
         CNLHS[i,i-1] = -0.5j*pm.sys.deltat*(0.5/pm.sys.deltax**2)
   return CNLHS

   
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
   pm.setup_space()

   v_ext = pm.space.v_ext


   # take external potential for initial guess
   waves = np.zeros((pm.sys.grid,pm.sys.NE))
   H = hamiltonian(pm, waves)
   den,eigf,eigv = groundstate(pm, H)
   
   # Calculate ground state density
   converged = False
   iteration = 1
   while (not converged):
      # Calculate new potentials form new orbitals
      H_new = hamiltonian(pm, eigf)
    
      # Stability mixing
      H_new = (1-pm.hf.nu)*H + pm.hf.nu*H_new 
      
      # Diagonalise Hamiltonian
      den_new, eigf, eigv = groundstate(pm, H_new)
      
      dn = np.sum(np.abs(den-den_new))*pm.sys.deltax
      converged = dn < pm.hf.con
      s = 'HF: dn = {:+.3e}, iter = {}'\
          .format(dn, iteration)
      pm.sprint(s, 1, newline=False)
 
      iteration += 1
      H = H_new
      den = den_new
   print()
   
   # Calculate ground state energy
   E_HF = total_energy(pm, eigf, eigv)
   pm.sprint('HF: hartree-fock energy = {}'.format(E_HF.real), 1, newline=True)
   
   results = rs.Results()
   results.add(E_HF,'gs_hf_E')
   results.add(den,'gs_hf_den')

   if pm.hf.save_eig:
       results.add(eigf.T, 'gs_hf_eigf')
       results.add(eigv, 'gs_hf_eigv')

   if pm.run.save:
      results.save(pm)

   if pm.run.time_dependence:
       for j in range(1, pm.sys.imax):
         string = 'HF: evolving through real time: t = {:.4f}'.format(j*pm.sys.deltat)
         pm.sprint(string,1,newline=False)

         n_t,Psi = CrankNicolson(pm, v_ks_t,Psi,n_t,j)
         if j != pm.sys.imax-1:
            v_ks_t[j+1,:] = v_ext[:]+hartree_potential(pm, n_t[j,:])+VXC(pm, n_t[j,:])
         current[j,:] = CalculateCurrentDensity(pm, n_t,j)
         v_xc_t[j,:] = VXC(pm, n_t[j,:])


       #TODO: 
       # implement stencil
       # put crank-nicholson
       # put current density

       # Output results
       #results.add(v_ks_t, 'td_hf_vks')
       #results.add(v_xc_t, 'td_hf_vxc')
       results.add(n_t, 'td_hf_den')
       results.add(current, 'td_hf_cur')

       if pm.run.save:
          l = ['td_lda_vks','td_lda_vxc','td_lda_den','td_lda_cur']
          results.save(pm, list=l)
 
   return results

