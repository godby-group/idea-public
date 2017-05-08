"""Computes ground-state charge density in the Hartree-Fock approximation. 

The code outputs the ground-state charge density, the energy of the system and
the Hartree-Fock orbitals. 
Can perform adiabatic time-dependent Hartree-Fock calculations.
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
from . import RE_Utilities


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
    return n.real


def hamiltonian(pm, wfs, perturb=False):
    r"""Compute HF Hamiltonian

    Computes HF Hamiltonian from a given set of single-particle states

    parameters
    ----------
    wfs  array_like
      single-particle states
    perturb: bool
      If True, add perturbation to external potential (for time-dep. runs)

    returns array_like
         Hamiltonian matrix
    """
    sd = pm.space.second_derivative
    sd_ind = pm.space.second_derivative_indices

    # construct kinetic energy
    K = -0.5*sps.diags(sd, sd_ind, shape=(pm.sys.grid,pm.sys.grid), format='csr', dtype=complex)

    # construct external and hartree potential
    n = electron_density(pm, wfs)
    V = pm.space.v_ext + hartree(pm,n)
    if perturb:
      V += pm.space.v_pert
    V = sps.diags(V, 0, shape=(pm.sys.grid,pm.sys.grid), format='csr', dtype=complex)
    
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
   n: array_like
      Time dependent density of the system indexed as n[time_index][space_index]
   j: int
      time index of current iteration

   returns
   -------
   J: array_like
      Time dependent current density indexed as J[time_index][space_index]
   """
   J = RE_Utilities.continuity_eqn(pm.sys.grid,pm.sys.deltax,pm.sys.deltat,n[j,:],n[j-1,:])

   if pm.sys.im == 1:
      #TODO: Indices are messed up hear. Fix!
      for i in range(pm.sys.grid):
         for k in range(j+1):
            x = i*pm.sys.deltax-pm.sys.xmax
            J[j] -= abs(pm.sys.v_pert_im(x))*n[j,k]*pm.sys.deltax
   return J


def crank_nicolson_step(pm, waves, H):
   r"""Solves Crank Nicolson Equation
 
   .. math::

        \left(\mathbb{1} + i\frac{dt}{2}\right) \Psi(x,t+dt) = \left(\mathbb{1} - i \frac{dt}{2} H\right) \Psi(x,t)

   for :math:`\Psi(x,t+dt)`.

   parameters
   ----------
   total_td_density : array_like
      Time dependent density of the system indexed as total_td_density[time_index][space_index]

   returns array_like
      Time dependent current density indexed as current_density[time_index][space_index]
 
   """
   dH = 0.5J * pm.sys.deltat * H
   identity = np.eye(pm.sys.grid, dtype=np.complex)

   A = identity + dH
   Abar = identity - dH

   # solve for all single-particle states at once
   RHS = np.dot(Abar, waves[:, :pm.sys.NE])
   waves_new = spla.solve(A,RHS)

   return waves_new
 

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

   # take external potential for initial guess
   # (setting wave functions to zero yields V=V_ext)
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
      E_HF = total_energy(pm, eigf, eigv)
      s = 'HF: E = {:+.8f} Ha, dn = {:+.3e}, iter = {}'\
          .format(E_HF, dn, iteration)
      pm.sprint(s, 1, newline=False)
 
      iteration += 1
      H = H_new
      den = den_new
   pm.sprint()
   
   # Calculate ground state energy
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

      # Starting values for wave functions, density
      waves = eigf
      n_t = np.empty((pm.sys.imax, pm.sys.grid), dtype=np.float)
      n_t[0] = den
      current = np.empty((pm.sys.imax, pm.sys.grid), dtype=np.float)
      current[0] = 0.0

      for j in range(1, pm.sys.imax):
         s = 'HF: evolving through real time: t = {:.4f}'.format(j*pm.sys.deltat)
         pm.sprint(s,1,newline=False)

         waves = crank_nicolson_step(pm, waves, H)
         S = np.dot(waves.T.conj(), waves) * pm.sys.deltax
         orthogonal = np.allclose(S, np.eye(pm.sys.NE, dtype=np.complex),atol=1e-6) 
         if not orthogonal:
             pm.sprint("HF: Warning: Orthonormality of orbitals violated at iteration {}".format(j))

         den = electron_density(pm, waves)
         H = hamiltonian(pm, waves, perturb=True)

         n_t[j] = den
         current[j,:] = CalculateCurrentDensity(pm, n_t,j)

      pm.sprint()

      #TODO: 
      # put current density
      # don't store all psi's (also remove this from LDA)

      # Output results
      results.add(n_t, 'td_hf_den')
      results.add(current, 'td_hf_cur')

      if pm.run.save:
         l = ['td_hf_den','td_hf_cur']
         results.save(pm, list=l)
 
   return results
