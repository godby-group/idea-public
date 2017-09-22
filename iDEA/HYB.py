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


def hamiltonian(pm, eigf, density, alpha, perturb=False):

   # construct kinetic energy
   sd = pm.space.second_derivative
   sd_ind = pm.space.second_derivative_indices
   K = -0.5*sps.diags(sd, sd_ind, shape=(pm.sys.grid,pm.sys.grid), format='csr', dtype=complex).toarray()

   # construct external potential
   Vext = sps.diags(pm.space.v_ext, 0, shape=(pm.sys.grid,pm.sys.grid), format='csr', dtype=complex).toarray()
   if perturb == True:
       Vext += sps.diags(pm.space.v_pert, 0, shape=(pm.sys.grid,pm.sys.grid), format='csr', dtype=complex).toarray()

   # construct hartree potential
   Vh = iDEA.HF.hartree(pm, density)

   # construct LDA Vxc
   Vxc_LDA = iDEA.LDA.VXC(pm, density)

   # construct fock operator
   if alpha != 0.0:
      F = iDEA.HF.fock(pm, eigf)*pm.sys.deltax
   else:
      F = np.zeros((pm.sys.grid, pm.sys.grid), dtype=np.complex)

   # construct hybrid Vxc
   Vxc = alpha*F + (1-alpha)*np.diag(Vxc_LDA)

   # construct H
   H = K + Vext + np.diag(Vh) + Vxc
   return H, Vh, Vxc_LDA, F


def calc_with_alpha(pm, alpha, occupations):

   # Initialise self-consistency
   counter = 0
   convergence = 1.0
   H, Vh, Vxc_LDA, F = hamiltonian(pm, np.zeros((pm.sys.grid,pm.sys.grid)), np.zeros(pm.sys.grid), alpha)
   density, eigf, eigv = iDEA.HF.groundstate(pm, H)
   E = 0
   # Perform self-consistency
   while convergence > pm.hyb.tol and counter < pm.hyb.max_iter:

      # keep copies to check convergance
      density_old = copy.deepcopy(density)
      H_old =  copy.deepcopy(H)

      # Construct hybrid hamiltonian
      H, Vh, Vxc_LDA, F = hamiltonian(pm, eigf, density, alpha)

      # Mix for stability
      H = pm.hyb.mix*H + (1.0-pm.hyb.mix)*H_old

      # Solve single-particle SE
      density, eigf, eigv = iDEA.HF.groundstate(pm, H)

      # Get a list of occupied wavefunctions:
      occupied = eigf[:, :pm.sys.NE]

      # scale HOMO orbital by its occupation
      occupied[:, pm.sys.NE-1] = occupied[:, pm.sys.NE-1]*np.sqrt(pm.hyb.homo_occupation)

      # calculate density associated with occupied orbitals - this takes into account fractional occupation
      density = np.sum(occupied*occupied.conj(), axis=1).real

      # Test for convergance
      convergence = sum(abs(density - density_old))
      pm.sprint('HYB: computing GS density with alpha = {0}, convergence = {1}'.format(alpha, convergence), 1, newline=False)
      counter += 1

      # Calculate the total energy
      E_SYS = 0.0
      for i in range(0, pm.sys.NE):
         E_SYS += eigv[i]*occupations[i]
      E_H = 0.5*np.dot(Vh, density)*pm.sys.deltax
      E_F = 0.0
      for k in range(pm.sys.NE):
         orb = eigf[:,k]
         E_F -= 0.5 * np.dot(orb.conj().T, np.dot(F, orb)) * pm.sys.deltax * occupations[k]
      pm.lda.NE = 3
      E_LDA = iDEA.LDA.EXC(pm, density) - (np.dot(density, Vxc_LDA)*pm.sys.deltax)
      E = (E_SYS - E_H + alpha*E_F + (1.0 - alpha)*E_LDA).real

   pm.sprint('', 1, newline=True)
   pm.sprint('HYB: total energy = {0} converged in {1} iterations'.format(E, counter), 1, newline=True)
   return density, eigf, eigv, E


def save_results(pm, results, density, E, eigf, eigv, alpha):
   results.add(density,'gs_hyb{}_den'.format(alpha).replace('.','_'))
   results.add(E,'gs_hyb{}_E'.format(alpha).replace('.','_'))
   if pm.non.save_eig:
      results.add(eigf.T,'gs_hyb{}_eigf'.format(alpha).replace('.','_'))
      results.add(eigv,'gs_hyb{}_eigv'.format(alpha).replace('.','_'))
   if (pm.run.save):
      results.save(pm)


def optimal_alpha(pm, results, alphas, occupations):

   pm.sprint('HYB: Finding optimal value of alpha', 1, newline=True)

   # Running E(N) calculations:
   nElect = pm.sys.NE
   pm.sprint('HYB: Starting calculations for N electrons (N={})'.format(pm.sys.NE), 1, newline=True)
   energies, eigsHOMO = n_run(pm, results, alphas, occupations)

   # Running E(N-1) calculations:
   pm.sys.NE = nElect - 1
   pm.sprint('HYB: Starting calculations for N-1 electrons (N-1={})'.format(pm.sys.NE), 1, newline=True)
   energies_minus_one, eigsLUMO = n_minus_one_run(pm, results, alphas, occupations)

   # Save all results
   results.add(alphas,'gs_hyb_alphas')
   results.add(energies,'gs_hyb_enN')
   results.add(energies_minus_one,'gs_hyb_enNm')
   results.add(eigsLUMO,'gs_hyb_eigL')
   results.add(eigsHOMO,'gs_hyb_eigH')
   if (pm.run.save):
      results.save(pm)


def n_run(pm, results, alphas, occupations):
   energies  = np.array([])
   eigsHOMO  = np.array([])
   for alpha in alphas:
      density, eigf, eigv, energy = calc_with_alpha(pm, alpha, occupations)
      eigsHOMO = np.append(eigsHOMO, eigv[pm.sys.NE - 1])
      energies = np.append(energies, energy)
      save_results(pm, results, density, energy, eigf, eigv, alpha)
   return energies, eigsHOMO


def n_minus_one_run(pm, results, alphas, occupations):
   energies  = np.array([])
   eigsLUMO  = np.array([])
   for alpha in alphas:
      density, eigf, eigv, energy = calc_with_alpha(pm, alpha, occupations)
      eigsLUMO = np.append(eigsLUMO, eigv[pm.sys.NE])
      energies = np.append(energies, energy)
   return energies, eigsLUMO


def main(parameters):
   # Set up parameters
   pm = parameters
   pm.setup_space()
   results = rs.Results()
   pm.sprint('HYB: HOMO occupation = {}'.format(pm.hyb.homo_occupation), 1, newline=True)

   # Occupations of all the orbitals, only the last value should be fractional:
   occupations = np.ones(pm.sys.NE)

   # Run code to find optimal alpha - no fractional occupation in this part
   if pm.hyb.alpha == 'o':
       alphas = np.linspace(pm.hyb.alphas[0], pm.hyb.alphas[1], pm.hyb.alphas[2])
       optimal_alpha(pm, results, alphas, occupations)

   # Run code for one given alpha - fractional occupation only in here
   else:
       occupations[pm.sys.NE - 1] = pm.hyb.homo_occupation
       density, eigf, eigv, E = calc_with_alpha(pm, pm.hyb.alpha, occupations)
       save_results(pm, results, density, E, eigf, eigv, pm.hyb.alpha)

   if pm.run.time_dependence:

      # Starting values for wave functions, density
      if pm.hyb.alpha == 'o':
          raise ValueError('HYB: ERROR! Cannot optimise hybrid in time-dependence, please give a numerical value from alpha.')

      n_t = np.empty((pm.sys.imax, pm.sys.grid), dtype=np.float)
      n_t[0] = density
      H, Vh, Vxc_LDA, F = hamiltonian(pm, eigf, density, pm.hyb.alpha, perturb=False)

      for i in range(1, pm.sys.imax):
         string = 'HYB: evolving through real time: t = {:.4f}'.format(i*pm.sys.deltat)
         pm.sprint(string, 1, newline=False)

         eigf = iDEA.HF.crank_nicolson_step(pm, eigf, H)
         density = iDEA.HF.electron_density(pm, eigf)
         H, Vh, Vxc_LDA, F = hamiltonian(pm, eigf, density, pm.hyb.alpha, perturb=True)
         n_t[i] = density

      # Calculate the current density
      pm.sprint()
      current_density = iDEA.HF.calculate_current_density(pm, n_t)

      # Output results
      results.add(n_t, 'td_hyb_den')
      results.add(current_density, 'td_hyb_cur')

      if pm.run.save:
         l = ['td_hyb_den','td_hyb_cur']
         results.save(pm, list=l)

   return results
