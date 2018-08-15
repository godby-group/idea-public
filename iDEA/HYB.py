"""Computes ground-state and time-dependent charge density in the Hybrid
Hartree-Fock-LDA approximation.

The code outputs the ground-state charge density, the energy of the
system and the single-quasiparticle orbitals.
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import copy
import pickle
import numpy as np
import scipy as sp
import scipy.sparse as sps
import scipy.linalg as spla
from iDEA.input import Input
from . import results as rs
import iDEA.LDA
import iDEA.HF
import iDEA.NON


def hamiltonian(pm, eigf, density, alpha, occupations, perturb=False):
   r"""Compute HF Hamiltonian.

   Computes HYB Hamiltonian from a given set of single-particle states.

   .. math:: H_{\alpha}(x,y) = \delta(x-y)\hat{T} + \delta(x-y)v_{\text{ext}}(y) + \delta(x-y)v_{\text{H}}(y) + \alpha\Sigma_{\text{x}}(x,y) + (1-\alpha)\delta(x-y)v_{\text{xc}}^{\text{LDA}}(y)

   parameters
   ----------
   eigf  array_like
      single-particle states
   density  array_like
      electron density
   alpha  float
      HF-LDA mixing parameter (1 = all HF, 0 = all LDA)
   occupations: array_like
      orbital occupations
   perturb: bool
      If True, add perturbation to external potential (for time-dep. runs)

   returns array_like
      Hamiltonian matrix
    """

   # Construct kinetic energy.
   sd = pm.space.second_derivative
   sd_ind = pm.space.second_derivative_indices
   T = -0.5*sps.diags(sd, sd_ind, shape=(pm.sys.grid,pm.sys.grid), format='csr', dtype=complex).toarray()

   # Construct external potential.
   Vext = sps.diags(pm.space.v_ext, 0, shape=(pm.sys.grid,pm.sys.grid), format='csr', dtype=complex).toarray()
   if perturb:
      Vext += sps.diags(pm.space.v_pert, 0, shape=(pm.sys.grid,pm.sys.grid), format='csr', dtype=complex).toarray()

   # Construct Hartree potential.
   Vh = iDEA.HF.hartree(pm, density)

   # Construct LDA Vxc.
   if not pm.hyb.seperate:
      Vxc_LDA = iDEA.LDA.xc_potential(pm, density, pm.hyb.seperate)
   else:
      Vxc_LDA, Vx_LDA, Vc_LDA = iDEA.LDA.xc_potential(pm, density, pm.hyb.seperate)

   # Construct the Fock operator.
   if alpha != 0:
      F = iDEA.HF.fock(pm, eigf)
      F *= pm.space.delta
   else:
      F = np.zeros((pm.sys.grid, pm.sys.grid))

   # Construct hybrid Vxc.
   if not pm.hyb.seperate:
      Vxc = alpha*F + (1-alpha)*np.diag(Vxc_LDA)
   else:
      Vxc = alpha*F + (1-alpha)*np.diag(Vx_LDA) + np.diag(Vc_LDA)

   # Construct H.
   H = T + Vext + np.diag(Vh) + Vxc
   if not pm.hyb.seperate:
      return H, Vh, Vxc_LDA, F
   else:
      return H, Vh, Vxc_LDA, Vx_LDA, Vc_LDA, F

def calc_with_alpha(pm, alpha, occupations):
   r"""Calculate with given alpha.

   Perform hybrid calculation with given alpha.

   parameters
   ----------
   alpha  float
      HF-LDA mixing parameter (1 = all HF, 0 = all LDA)
   occupations: array_like
      orbital occupations

   returns density, eigf, eigv, E
      Hybrid Density, orbitals and total energy
    """

   # Initialise self-consistency.
   counter = 0
   convergence = 1.0
   H, Vh, Vxc_LDA, F = hamiltonian(pm, np.zeros((pm.sys.grid,pm.sys.grid)), np.zeros(pm.sys.grid), alpha, occupations)
   density, eigf, eigv = iDEA.HF.groundstate(pm, H)
   E = 0

   # Convergence boolean:
   converged = False

   # Perform self-consistency.
   while not converged and counter < pm.hyb.max_iter:
      # Keep copies to check convergence.
      density_old = copy.deepcopy(density)
      H_old =  copy.deepcopy(H)

      # Construct hybrid Hamiltonian.
      if not pm.hyb.seperate:
         H, Vh, Vxc_LDA, F = hamiltonian(pm, eigf, density, alpha, occupations)
      else:
         H, Vh, Vxc_LDA, Vx_LDA, Vc_LDA, F = hamiltonian(pm, eigf, density, alpha, occupations)

      # Mix for stability.
      H = pm.hyb.mix*H + (1.0-pm.hyb.mix)*H_old

      # Solve single-particle Shrodinger equation.
      density, eigf, eigv = iDEA.HF.groundstate(pm, H)

      if pm.sys.NE > 0:
         # Get a list of occupied wavefunctions.
         occupied = copy.deepcopy(eigf[:, :pm.sys.NE])

         # Scale HOMO orbital by its occupation.
         # Only this orbital can have fractional occupation.
         occupied[:,-1] = occupied[:,-1]*np.sqrt(occupations[-1])

         # Calculate density associated with occupied orbitals.
         # This takes into account fractional occupation.
         density = np.sum(occupied*occupied.conj(), axis=1).real

      # Test for convergance.
      convergence = sum(abs(density - density_old))
      converged = convergence < pm.hyb.tol and counter > 20

      pm.sprint('HYB: computing GS density with alpha = {:05.4f}, convergence = {:06.5e}'.format(alpha, convergence), 1, newline=False)
      counter += 1

      E_H = -0.5*np.dot(Vh, density)*pm.sys.deltax
      E_SYS = 0
      E_F = 0
      for i in range(pm.sys.NE):
         # Calculate system energy.
         E_SYS += eigv[i]*occupations[i]
         # Calculate exchange energy.
         orb = copy.deepcopy(eigf[:,i]) * np.sqrt(occupations[i])
         E_F -= 0.5 * np.dot(orb.conj().T, np.dot(F, orb)) * pm.sys.deltax

      # LDA energy:
      if pm.hyb.seperate:
          Exc_LDA, Ex_LDA, Ec_LDA = iDEA.LDA.xc_energy(pm, density)
          Evx_LDA = Ex_LDA - np.dot(density, Vx_LDA)*pm.sys.deltax
          Evc_LDA = Ec_LDA - np.dot(density, Vc_LDA)*pm.sys.deltax
          E = (E_SYS + E_H + alpha*E_F + (1.0 - alpha)*Evx_LDA + Evc_LDA).real
      else:
          Evxc_LDA = iDEA.LDA.xc_energy(pm, density) - np.dot(density, Vxc_LDA)*pm.sys.deltax
          E = (E_SYS + E_H + alpha*E_F + (1.0 - alpha)*Evxc_LDA).real

      # Calculate charges on grid.
      grid_charge = np.sum(density)*pm.sys.deltax

   # Print iteration values.
   pm.sprint('\nHYB: Total charge on grids: {:10.9f}'.format(grid_charge), 1, newline=True)
   pm.sprint('HYB: total energy = {0} converged in {1} iterations'.format(E, counter), 1, newline=True)
   pm.sprint('HYB: HOMO-LUMO gap = {0}\n'.format(eigv[pm.sys.NE]-eigv[pm.sys.NE-1]), 1, newline=True)
   return density, eigf, eigv, E


def save_results(pm, results, density, E, eigf, eigv, alpha):
   r"""Saves hybrid results to outputs directory
   """
   results.add(density,'gs_hyb{:05.3f}_den'.format(alpha).replace('.','_'))
   results.add(E,'gs_hyb{:05.3f}_E'.format(alpha).replace('.','_'))
   results.add(eigf.T,'gs_hyb{:05.3f}_eigf'.format(alpha).replace('.','_'))
   results.add(eigv,'gs_hyb{:05.3f}_eigv'.format(alpha).replace('.','_'))
   if pm.run.save:
      results.save(pm)


def optimal_alpha(pm, results, alphas, occupations):
   r"""Calculate optimal alpha.

   Calculate over range of alphas to determine optimal alpha.

   parameters
   ----------
   results  Results Object
      object to add results to
   alphas: array_like
      range of alphas to use
   occupations: array_like
      orbital occupations
   """
   pm.sprint('HYB: Finding optimal value of alpha', 1, newline=True)

   # Run E(N) calculations.
   nElect = pm.sys.NE
   pm.sprint('HYB: Starting calculations for N electrons (N={})'.format(pm.sys.NE), 1, newline=True)
   energies, eigsHOMO = n_run(pm, results, alphas, occupations)

   # Run E(N-1) calculations.
   pm.sys.NE = nElect - 1
   pm.sprint('HYB: Starting calculations for N-1 electrons (N-1={})'.format(pm.sys.NE), 1, newline=True)
   energies_minus_one, eigsLUMO = n_minus_one_run(pm, results, alphas, occupations)

   # Save all results.
   results.add(alphas,'gs_hyb_alphas')
   results.add(energies,'gs_hyb_enN')
   results.add(energies_minus_one,'gs_hyb_enNm')
   results.add(eigsLUMO,'gs_hyb_eigL')
   results.add(eigsHOMO,'gs_hyb_eigH')
   if pm.run.save:
      results.save(pm)


def n_run(pm, results, alphas, occupations):
   r"""Calculate for :math:`N` electron run.

   Calculate total energy and HOMO eigenvalue of N electron system.

   parameters
   ----------
   results  Results Object
      object to add results to
   alphas: array_like
      range of alphas to use
   occupations: array_like
      orbital occupations
   """
   alphas_length = alphas.shape[0]
   eigsHOMO  = np.empty(alphas_length)
   energies  = np.empty(alphas_length)
   for i in range(alphas_length):
      density, eigf, eigv, energy = calc_with_alpha(pm, alphas[i], occupations)
      eigsHOMO[i] = eigv[pm.sys.NE - 1]
      energies[i] = energy
      save_results(pm, results, density, energy, eigf, eigv, alphas[i])
   return energies, eigsHOMO


def n_minus_one_run(pm, results, alphas, occupations):
   r"""Calculate for :math:`N-1` electron run.

   Calculate total energy and LUMO eigenvalue of :math:`N-1` electron
   system.

   parameters
   ----------
   results  Results Object
      object to add results to
   alphas: array_like
      range of alphas to use
   occupations: array_like
      orbital occupations
   """
   alphas_length = alphas.shape[0]
   eigsLUMO  = np.empty(alphas_length)
   energies  = np.empty(alphas_length)
   for i in range(alphas_length):
      density, eigf, eigv, energy = calc_with_alpha(pm, alphas[i], occupations)
      eigsLUMO[i] = eigv[pm.sys.NE]
      energies[i] = energy
   return energies, eigsLUMO

def fractional_run(pm, results, occupations, fractions):
   energies = np.array([])
   eigsHOMO = np.array([])
   eigsLUMO = np.array([])
   pm.sprint('\nHYB: Running fractional numbers of electrons from {} to {}\n'.format(fractions[0], fractions[-1]), 1, newline=True)
   for num_electrons in fractions:
      pm.sprint('HYB: Current total number of electrons = {:05.4f}'.format(num_electrons), 1, newline=True)
      pm.sys.NE = int(np.ceil(num_electrons))
      if pm.sys.NE == 0:
         pm.sys.NE = 1
         occupations = np.zeros(pm.sys.NE)
      else:
         occupations = np.ones(pm.sys.NE)
         occupations[-1] = num_electrons - int(np.ceil(num_electrons)) + 1
      # Run a normal calculation.
      density, eigf, eigv, energy = calc_with_alpha(pm, pm.hyb.alpha, occupations)
      eigsHOMO = np.append(eigsHOMO, eigv[pm.sys.NE - 1])
      eigsLUMO = np.append(eigsLUMO, eigv[pm.sys.NE])
      energies = np.append(energies, energy)
   results.add(fractions,'gs_hyb_frac{:05.3f}'.format(pm.hyb.alpha).replace('.','_'))
   results.add(eigsHOMO, 'gs_hyb_frac{:05.3f}_HOMO'.format(pm.hyb.alpha).replace('.','_'))
   results.add(eigsLUMO, 'gs_hyb_frac{:05.3f}_LUMO'.format(pm.hyb.alpha).replace('.','_'))
   results.add(energies, 'gs_hyb_frac{:05.3f}_en'.format(pm.hyb.alpha).replace('.','_'))
   if (pm.run.save):
      results.save(pm)


def main(parameters):
   r"""Performs Hybrid calculation.

   parameters
   ----------
   parameters : object
      Parameters object

   returns object
      Results object
   """
   # Set up parameters.
   pm = parameters
   pm.setup_space()
   results = rs.Results()

   # Choose type of LDA to be run.
   if pm.hyb.seperate:
      pm.lda.NE = 'heg'
   else:
      pm.lda.NE = 3

   # Occupations of all the orbitals.
   # Only the last value should be fractional.
   occupations = np.ones(pm.sys.NE)

   # Run code to find optimal alpha.
   # No fractional occupation in this part.
   if pm.hyb.functionality == 'o':
      alphas = np.linspace(pm.hyb.of_array[0], pm.hyb.of_array[1], pm.hyb.of_array[2])
      optimal_alpha(pm, results, alphas, occupations)

   # Run code to get fractional numbers of electrons.
   elif pm.hyb.functionality == 'f':
      fractions = np.linspace(pm.hyb.of_array[0], pm.hyb.of_array[1], pm.hyb.of_array[2])
      #fractions = np.array([1.0, 0.9])
      fractional_run(pm, results, occupations, fractions)

   # Run code for one given alpha.
   elif pm.hyb.functionality == 'a':
      occupations[pm.sys.NE - 1] = 1.0
      density, eigf, eigv, E = calc_with_alpha(pm, pm.hyb.alpha, occupations)
      save_results(pm, results, density, E, eigf, eigv, pm.hyb.alpha)

   # Print an error message.
   else:
      pm.sprint('HYB: Functionality chosen is not valid. Please choose from a, o, and f.', 1, newline=True)

   return results
