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


def hamiltonian(pm, eigf, density, alpha):

   # construct kinetic energy
   sd = pm.space.second_derivative
   sd_ind = pm.space.second_derivative_indices
   K = -0.5*sps.diags(sd, sd_ind, shape=(pm.sys.grid,pm.sys.grid), format='csr', dtype=complex).toarray()

   # construct external potential
   Vext = sps.diags(pm.space.v_ext, 0, shape=(pm.sys.grid,pm.sys.grid), format='csr', dtype=complex).toarray()

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


def calc_with_alpha(pm, alpha):

   # Initialise self-consistency
   counter = 0
   convergence = 1.0
   H, Vh, Vxc_LDA, F = hamiltonian(pm, np.zeros((pm.sys.grid,pm.sys.grid)), np.zeros(pm.sys.grid), alpha)
   density, eigf, eigv = iDEA.HF.groundstate(pm, H)

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

      # Test for convergance
      convergence = sum(abs(density - density_old))
      pm.sprint('HYB: computing ground-state density with alpha = {0}, convergence = {1}'.format(alpha, convergence), 1, newline=False)
      counter += 1

      # Calculate the total energy
      E_SYS = 0.0
      for i in range(0, pm.sys.NE):
         E_SYS += eigv[i]
      E_H = 0.5*np.dot(Vh, density)*pm.sys.deltax
      E_F = 0.0
      for k in range(pm.sys.NE):
         orb = eigf[:,k]
         E_F -= 0.5 * np.dot(orb.conj().T, np.dot(F, orb)) * pm.sys.deltax
      pm.lda.NE = 3
      E_LDA = iDEA.LDA.EXC(pm, density) - (np.dot(density, Vxc_LDA)*pm.sys.deltax)
      E = (E_SYS - E_H + alpha*E_F + (1.0 - alpha)*E_LDA).real

   return density, eigf, eigv, E


def save_results(pm, results, density, E, eigf, eigv, alpha):
   results.add(density,'gs_hyb_den{}'.format(alpha))
   results.add(E,'gs_hyb_E{}'.format(alpha))
   if pm.non.save_eig:
      results.add(eigf.T,'gs_hyb_eigf{}'.format(alpha))
      results.add(eigv,'gs_hyb_eigv{}'.format(alpha))
   if (pm.run.save):
      results.save(pm)


def optimal_alpha(pm, results, alphas):
   # Running E(N) calculations:
   nElect = pm.sys.NE
   energies, eigsHOMO = n_run(pm, results, alphas)

   # Running E(N-1) calculations:
   pm.sys.NE = nElect - 1
   energies_minus_one, eigsLUMO = n_minus_one_run(pm, results, alphas)

   # Save all results
   results.add(alphas,'gs_hyb_alphas')
   results.add(energies,'gs_hyb_enN')
   results.add(energies_minus_one,'gs_hyb_enNm')
   results.add(eigsLUMO,'gs_hyb_eigL')
   results.add(eigsHOMO,'gs_hyb_eigH')
   if (pm.run.save):
      results.save(pm)


def n_run(pm, results, alphas):
   energies  = np.array([])
   eigsHOMO  = np.array([])
   print (alphas)
   for alpha in alphas:
      density, eigf, eigv, energy = calc_with_alpha(pm, alpha)
      eigsHOMO = np.append(eigsHOMO, eigv[pm.sys.NE - 1])
      energies = np.append(energies, energy)
      save_results(pm, results, density, energy, eigf, eigv, alpha)
   return energies, eigsHOMO


def n_minus_one_run(pm, results, alphas):
   energies  = np.array([])
   eigsLUMO  = np.array([])
   for alpha in alphas:
      density, eigf, eigv, energy = calc_with_alpha(pm, alpha)
      eigsLUMO = np.append(eigsLUMO, eigv[pm.sys.NE])
      energies = np.append(energies, energy)
   return energies, eigsLUMO


def main(parameters):
   # Set up parameters
   pm = parameters
   pm.setup_space()
   results = rs.Results()

   # Run code to find optimal alpha
   if pm.hyb.alpha == 'o':
       alphas = np.linspace(pm.hyb.alphas[0], pm.hyb.alphas[1], pm.hyb.alphas[2])
       optimal_alpha(pm, results, alphas)

   # Run code for one given alpha
   else:
       density, eigf, eigv, E = calc_with_alpha(pm, pm.hyb.alpha)

   return results
