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
from . import RE_cython

def read_input_density(pm, approx):
    r"""Reads in the electron density that was calculated using the selected
    approximation.

    parameters
    ----------
    pm : object
        Parameters object
    approx : string
        The approximation used to calculate the electron density

    returns array_like
        Array of the ground-state electron density from the
        approximation, indexed as density_approx[space_index]
    """
    density_approx = np.zeros((pm.space.npt), dtype=np.float)
    name = 'gs_{}_den'.format(approx)
    density_approx[:] = rs.Results.read(name, pm)
    return density_approx


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
   eigv, eigf = spla.eigh(H)
   eigf = eigf/ np.sqrt(pm.sys.deltax)
   n = electron_density(pm,eigf)

   return n, eigf, eigv

def main(parameters, approx):
   r"""Reverse-egineers the density to find the corresponding 
   exact form of the local correlation potential within 
   Hartree-Fock-Kohn-Sham theory

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

   density_approx = read_input_density(pm, approx)
   mu = copy.copy(pm.hfks.mu)
   p = copy.copy(pm.hfks.p)

   # Calculate ground state density
   converged = False
   iteration = 1
   while (not converged):

      # Calculate new potentials form new orbitals
      H_new = hamiltonian(pm, eigf)
      v_c[:] += mu*(den[:]**p-density_approx[:]**p)
      Vc = sps.diags(v_c, 0, shape=(pm.sys.grid,pm.sys.grid), format='csr', dtype=complex)
      H_new += Vc.toarray()

      # Diagonalise Hamiltonian
      den_new, eigf, eigv = groundstate(pm, H_new)

      dn = np.sum(np.abs(den-density_approx))*pm.sys.deltax
      converged = dn < pm.hfks.con

      iteration += 1
      H = H_new
      den = den_new
      string = 'HFKS: density error = {}'.format(dn)
      pm.sprint(string, 1, newline=False)

   pm.sprint()

   results = rs.Results()
   results.add(v_c,'gs_{}hfks_vc'.format(approx))
   results.add(den,'gs_{}hfks_den'.format(approx))
   results.add(hartree(pm, den),'gs_{}hfks_vh'.format(approx))
   results.add(fock(pm, eigf),'gs_{}hfks_F'.format(approx))

   if pm.hf.save_eig:
       results.add(eigf.T, 'gs_{}hfks_eigf'.format(approx))
       results.add(eigv, 'gs_{}hfks_eigv'.format(approx))

   if pm.run.save:
      results.save(pm)

   return results
