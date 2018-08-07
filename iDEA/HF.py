"""Computes ground-state and time-dependent charge density in the Hartree-Fock approximation.
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
from . import RE_cython


def hartree(pm, density):
   r"""Computes Hartree potential for a given density

   .. math:: v_{\text{H}}(x) = \int \! n(y)u(x,y) \, \mathrm{d}y

   parameters
   ----------
   pm : object
        Parameters object
   density : array_like
        given density

   returns array_like
        Hartree potential
   """
   return np.dot(pm.space.v_int, density)*pm.space.delta


def fock(pm, eigf):
   r"""Constructs Fock operator from a set of orbitals

   .. math:: \Sigma_{\text{x}}(x,y) = - \sum_{j=1}^{N} \varphi_{j}^{*}(y) \varphi_{j}(x) u(x,y)

   where u(x,y) denotes the appropriate Coulomb interaction.

   parameters
   ----------
   pm : object
      Parameters object
   eigf : array_like
      Eigenfunction orbitals

   returns
   -------
   F: array_like
      Fock matrix
   """
   F = np.zeros((pm.sys.grid,pm.sys.grid), dtype='complex')
   for i in range(pm.sys.NE):
       orb = eigf[:,i]
       F -= np.tensordot(orb, orb.conj(), axes=0)
   F = F * pm.space.v_int
   return F


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


def hamiltonian(pm, wfs, perturb=False):
   r"""Compute HF Hamiltonian

   Computes HF Hamiltonian from a given set of single-particle states

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
   T = -0.5*sps.diags(sd, sd_ind, shape=(pm.sys.grid,pm.sys.grid), format='csr', dtype=complex).toarray()

   # Construct potentials
   if perturb:
      Vext = np.diag(pm.space.v_ext + pm.space.v_pert)
   else:
      Vext = np.diag(pm.space.v_ext)
   Vh = np.diag(hartree(pm, electron_density(pm, wfs)))
   if pm.hf.fock == 1:
      F = fock(pm,wfs)
   else:
      F = np.zeros(shape=Vh.shape)

   # Construct H
   H = T + Vext + Vh + F*pm.space.delta
   return H


def groundstate(pm, H):
   r"""Diagonalises Hamiltonian H

   .. math:: \hat{H}\varphi_{j} = \varepsilon_{j}\varphi_{j}

   parameters
   ----------
   pm: object
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
   string = 'HF: calculating current density'
   pm.sprint(string, 1, newline=True)
   for i in range(1, pm.sys.imax):
      string = 'HF: t = {:.5f}'.format(i*pm.sys.deltat)
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

   # Take external potential for initial guess
   # (setting wave functions to zero yields V=V_ext)
   waves = np.zeros((pm.sys.grid,pm.sys.NE), dtype=np.complex)
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

      # save for next iteration
      iteration += 1
      H = H_new
      den = den_new
   pm.sprint()

   # Calculate ground state energy
   pm.sprint('HF: hartree-fock energy = {}'.format(E_HF.real), 1, newline=True)

   # Construct results object
   results = rs.Results()
   results.add(E_HF,'gs_hf_E')
   results.add(-eigv[pm.sys.NE-1],'gs_hf_IP')
   results.add(-eigv[pm.sys.NE],'gs_hf_AF')
   results.add(eigv[pm.sys.NE]-eigv[pm.sys.NE-1],'gs_hf_GAP')
   results.add(den,'gs_hf_den')
   results.add(hartree(pm, den),'gs_hf_vh')
   results.add(fock(pm, eigf),'gs_hf_F')
   results.add(eigf.T, 'gs_hf_eigf')
   results.add(eigv, 'gs_hf_eigv')

   # Save results
   if pm.run.save:
      results.save(pm)

   if pm.run.time_dependence:

      # Starting values for wave functions, density
      waves[:, :pm.sys.NE] = eigf[:, :pm.sys.NE]
      n_t = np.empty((pm.sys.imax, pm.sys.grid), dtype=np.float)
      F_t = np.empty((pm.sys.imax, pm.sys.grid, pm.sys.grid), dtype=np.complex)
      Vh_t = np.empty((pm.sys.imax, pm.sys.grid) , dtype=np.float)
      n_t[0] = den
      F_t[0,:,:] = fock(pm, waves)
      Vh_t[0,:] = hartree(pm, den)

      # Perform time evolution
      for i in range(1, pm.sys.imax):
         string = 'HF: evolving through real time: t = {:.4f}'.format(i*pm.sys.deltat)
         pm.sprint(string, 1, newline=False)
         H = hamiltonian(pm, waves, perturb=True)
         waves[:, :pm.sys.NE] = crank_nicolson_step(pm, waves, H)
         S = np.dot(waves.T.conj(), waves) * pm.sys.deltax
         orthogonal = np.allclose(S, np.eye(pm.sys.NE, dtype=np.complex),atol=1e-6)
         if not orthogonal:
             pm.sprint("HF: Warning: Orthonormality of orbitals violated at iteration {}".format(i+1))
         den = electron_density(pm, waves)
         n_t[i] = den
         F_t[i,:,:] = fock(pm, waves)
         Vh_t[i,:] = hartree(pm, den)
      pm.sprint()

      # Calculate the current density
      current_density = calculate_current_density(pm, n_t)

      # Output results
      pm.sprint('HF: saving quantities...', 1, newline=True)
      results.add(n_t, 'td_hf_den')
      results.add(F_t, 'td_hf_F')
      results.add(Vh_t, 'td_hf_vh')
      results.add(current_density, 'td_hf_cur')
      if pm.run.save:
         l = ['td_hf_den','td_hf_cur', 'td_hf_F', 'td_hf_vh']
         results.save(pm, list=l)

   return results
