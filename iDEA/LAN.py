"""Computes time-dependent charge density of a system using the Landauer-Buttiker approximation. The code outputs the time-dependent charge and current density of the system.

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
from . import RE_cython


def electron_density(pm, orbitals):
   r"""Compute density for given orbitals

   parameters
   ----------
   pm : object
         Parameters object
   orbitals: array_like
         Array of properly normalised orbitals

   returns
   -------
   n: array_like
         electron density
   """
   occupied = orbitals[:, :pm.sys.NE]
   n = np.sum(occupied*occupied.conj(), axis=1)
   return n.real


# Function to read input
def ReadInput(pm,approx):
    r"""Reads in the ground-state Kohn-Sham (KS) potential

    parameters
    ----------
    pm : object
        Parameters object
    approx : string
        The approximation used to calculate the potential

    returns array_like
        1D array of the ground-state KS potential from the
        approximation, indexed as density_approx[space_index]
    """

    V = np.zeros(pm.sys.grid,dtype='complex')
    name = 'gs_{}_vks'.format(approx)
    data = rs.Results.read(name, pm)
    V[:] = data
    return V

def hamiltonian(pm, wfs, V_gs, perturb=False):
   r"""Compute LAN Hamiltonian

   Computes LAN Hamiltonian from a given set of single-particle states

   parameters
   ----------
   pm : object
         Parameters object
   wfs  array_like
         single-particle states
   perturb: bool
         If True, add perturbation to external potential (for time-dep. runs)

   returns array_like
         Hamiltonian matrix
   """
   # Construct kinetic energy
   sd = pm.space.second_derivative
   sd_ind = pm.space.second_derivative_indices
   K = -0.5*sps.diags(sd, sd_ind, shape=(pm.sys.grid,pm.sys.grid), format='csr', dtype=complex).toarray()

  # for i in range(pm.sys.grid):
  #     V[i,i] = V_gs[i]

   # Construct potentials
   if perturb:
      V = np.diag(V_gs+pm.space.v_pert)
   else:
      V = np.diag(V_gs)

   # Construct H
   H = K + V
   return H


def groundstate(pm, H):
   r"""Diagonalises Hamiltonian H

    .. math:: H = K + V + F \\
              H \psi_{i} = E_{i} \psi_{i}

   parameters
   ----------
   pm : object
         Parameters object
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
   # Solve eigen equation
   eigv, eigf = spla.eigh(H)
   eigf = eigf/ np.sqrt(pm.sys.deltax)

   # Calculate density
   n = electron_density(pm,eigf)
   return n, eigf, eigv


def calculate_current_density(pm, density):
   r"""Calculates the current density from the time-dependent
   (and ground-state) electron density by solving the continuity equation.

   .. math::

        \frac{\partial n}{\partial t} + \nabla \cdot j = 0

   parameters
   ----------
   pm : object
         Parameters object
   density : array_like
         2D array of the time-dependent density, indexed as
         density[time_index,space_index]

   returns array_like
         2D array of the current density, indexed as
         current_density[time_index,space_index]
   """
   pm.sprint('', 1, newline=True)
   current_density = np.zeros((pm.sys.imax,pm.sys.grid), dtype=np.float)
   string = 'LAN: calculating current density'
   pm.sprint(string, 1, newline=True)
   for i in range(1, pm.sys.imax):
      string = 'LAN: t = {:.5f}'.format(i*pm.sys.deltat)
      pm.sprint(string, 1, newline=False)
      J = np.zeros(pm.sys.grid, dtype=np.float)
      J = RE_cython.continuity_eqn(pm, density[i,:], density[i-1,:])
      current_density[i,:] = J[:]
   pm.sprint('', 1, newline=True)
   return current_density


def crank_nicolson_step(pm, waves, H):
   r"""Solves Crank Nicolson Equation

   .. math::

        \left(\mathbb{1} + i\frac{dt}{2} H\right) \Psi(x,t+dt) = \left(\mathbb{1} - i \frac{dt}{2} H\right) \Psi(x,t)

   for :math:`\Psi(x,t+dt)`.

   parameters
   ----------
   pm : object
         Parameters object
   total_td_density : array_like
         Time dependent density of the system indexed as total_td_density[time_index][space_index]

   returns array_like
         Time dependent current density indexed as current_density[time_index][space_index]

   """
   # Construct matrices
   dH = 0.5J * pm.sys.deltat * H
   identity = np.eye(pm.sys.grid, dtype=np.complex)
   A = identity + dH
   Abar = identity - dH

   # Solve for all single-particle states at once
   RHS = np.dot(Abar, waves[:, :pm.sys.NE])
   waves_new = spla.solve(A,RHS)
   return waves_new


def main(parameters):
   r"""Performs Landauer-Buttiker calculation

   parameters
   ----------
   parameters : object
         Parameters object

   returns object
         Results object
   """
   pm = parameters
   pm.setup_space()

   # Read the input ground-state KS potential
   V = ReadInput(pm,pm.lan.start)

   # Calculate ground state density
   waves = np.zeros((pm.sys.grid,pm.sys.NE), dtype=np.complex)
   H = hamiltonian(pm, waves, V)
   den,eigf,eigv = groundstate(pm, H)

   # Construct results object
   results = rs.Results()
   results.add(den,'gs_hf_den')

   # Save results
   if pm.run.save:
      results.save(pm)

   if pm.run.time_dependence:

      # Starting values for wave functions, density
      waves[:, :pm.sys.NE] = eigf[:, :pm.sys.NE]
      n_t = np.empty((pm.sys.imax, pm.sys.grid), dtype=np.float)
      n_t[0] = den

      H = hamiltonian(pm, waves, V, perturb=True)

      # Perform time evolution
      for i in range(1, pm.sys.imax):
         string = 'LAN: evolving through real time: t = {:.4f}'.format(i*pm.sys.deltat)
         pm.sprint(string, 1, newline=False)
         waves[:, :pm.sys.NE] = crank_nicolson_step(pm, waves, H)
         S = np.dot(waves.T.conj(), waves) * pm.sys.deltax
         orthogonal = np.allclose(S, np.eye(pm.sys.NE, dtype=np.complex),atol=1e-6)
         if not orthogonal:
             pm.sprint("LAN: Warning: Orthonormality of orbitals violated at iteration {}".format(i+1))
         den = electron_density(pm, waves)
         n_t[i] = den
      pm.sprint()

      # Calculate the current density
      current_density = calculate_current_density(pm, n_t)

      # Output results
      pm.sprint('LAN: saving quantities...', 1, newline=True)
      results.add(n_t, 'td_lan_den')
      results.add(current_density, 'td_lan_cur')
      if pm.run.save:
         l = ['td_lan_den','td_lan_cur']
         results.save(pm, list=l)

   return results
