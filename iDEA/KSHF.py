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

# Function to read inputs -- needs some work!
def ReadInput(approx,pm):
   n = np.zeros(pm.sys.grid)
   # Read in the ground-state first

   name = 'gs_{}_den'.format(approx)
   data = rs.Results.read(name, pm)
   
   n[:] = data

   return n


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
   v_c =  np.zeros(pm.sys.grid)
   H = hamiltonian(pm, waves)
   den,eigf,eigv = groundstate(pm, H)

   den_ext = ReadInput('ext',pm)

   mu = 1.0
   p = 0.05
   
   # Calculate ground state density
   converged = False
   iteration = 1
   while (not converged):
      # Calculate new potentials form new orbitals
      H_new = hamiltonian(pm, eigf)
      
      v_c[:] += mu*(den[:]**p-den_ext[:]**p)

      Vc = sps.diags(v_c, 0, shape=(pm.sys.grid,pm.sys.grid), format='csr', dtype=complex)
      H_new += Vc.toarray()

      # Diagonalise Hamiltonian
      den_new, eigf, eigv = groundstate(pm, H_new)
      
      dn = np.sum(np.abs(den-den_ext))*pm.sys.deltax
      converged = dn < pm.kshf.con
 
      iteration += 1
      H = H_new
      den = den_new
      string = 'REV: cost= {}'.format(dn)
      pm.sprint(string,1,newline=False)
   pm.sprint()
   
   results = rs.Results()
   results.add(v_c,'gs_kshf_cor')
   results.add(den,'gs_kshf_den')

   if pm.hf.save_eig:
       results.add(eigf.T, 'gs_kshf_eigf')
       results.add(eigv, 'gs_kshf_eigv')

   if pm.run.save:
      results.save(pm)

