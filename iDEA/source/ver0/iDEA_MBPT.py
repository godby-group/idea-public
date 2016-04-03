######################################################################################
# Name: Many Body Pertubation Theory (IN-DEVELOPMENT)                                #
######################################################################################
# Author(s): Jack Wetherell                                                          #
######################################################################################
# Description:                                                                       #
# Computes ground-state density of a system using the fully self-consistent GW       #
# approximation                                                                      #
#                                                                                    #
#                                                                                    #
######################################################################################
# Notes:                                                                             #
#                                                                                    #
#                                                                                    #
#                                                                                    #
######################################################################################

# Library imports
import math
import pickle
import sprint
import numpy as np
import scipy as sp
import parameters as pm
import scipy.sparse as sps
import scipy.sparse.linalg as spla

# Struct to define space-time grid
class SpaceTime:
   def __init__(self, tau_max, tau_N, x_grid):
      self.tau_grid = np.linspace(-tau_max,tau_max,tau_N)
      self.x1_grid = np.linspace(-pm.xmax,pm.xmax,x_grid)
      self.x2_grid = np.linspace(-pm.xmax,pm.xmax,x_grid)

# Function to construct the kinetic energy K
def constructK():
   xgrid = np.linspace(-pm.xmax,pm.xmax,pm.grid)
   K = -0.5*sps.diags([1, -2, 1],[-1, 0, 1], shape=(pm.grid,pm.grid), format='csr')/(pm.deltax**2)
   return K

# Function to construct the potential V
def constructV(td):
   xgrid = np.linspace(-pm.xmax,pm.xmax,pm.grid)
   Vdiagonal = []
   if(td == 0):
      for i in range(0,len(xgrid)):
         Vdiagonal.append(pm.well(xgrid[i]))
   if(td == 1):
      for i in range(0,len(xgrid)):
         Vdiagonal.append(pm.well(xgrid[i]) + pm.petrb(xgrid[i]))
   V = sps.spdiags(Vdiagonal, 0, pm.grid, pm.grid, format='csr')
   return V

# Function to construct the non-interacting Green's function G0
def non_interacting_greens_function(st, occupied, occupied_energies, empty, empty_energies):
   G0 = []
   for i in range(0,len(st.tau_grid)/2):
      tau = st.tau_grid[i]
      G0_tau = np.zeros((len(st.x1_grid),len(st.x2_grid)), dtype=np.cfloat)
      for j in range(0,len(st.x1_grid)):
         for k in range(0,len(st.x2_grid)):
            for n in range(0,len(occupied)):
               G0_tau[j][k] = G0_tau[j][k] + occupied[n][j]*np.conjugate(occupied[n][k])*math.exp(occupied_energies[n]*tau)
      G0_tau = 1.0j * G0_tau
      G0.append(G0_tau)
      string = 'MBPT: computing non-interacting Green\'s function in imaginary time, ' + 'tau = ' + str(tau)
      sprint.sprint(string,1,1,pm.msglvl)
   for i in range(len(st.tau_grid)/2,len(st.tau_grid)):
      tau = st.tau_grid[i]
      G0_tau = np.zeros((len(st.x1_grid),len(st.x2_grid)), dtype=np.cfloat)
      for j in range(0,len(st.x1_grid)):
         for k in range(0,len(st.x2_grid)):
            for n in range(0,len(empty)):
               G0_tau[j][k] = G0_tau[j][k] + empty[n][j]*np.conjugate(empty[n][k])*math.exp(empty_energies[n]*tau)
      G0_tau = -1.0j * G0_tau
      G0.append(G0_tau)
      string = 'MBPT: computing non-interacting Green\'s function in imaginary time, ' + 'tau = ' + str(tau)
      sprint.sprint(string,1,1,pm.msglvl)
   print
   return G0

# Main function
def main():

   # Construct the kinetic energy
   K = constructK()

   # Construct the potential
   V = constructV(0)

   # Constuct the Hamiltonian
   H = K + V

   # Compute all wavefunctions
   print 'MBPT: computing occupied and empty eigenfunctions of non-interacting hamiltonian'
   solution = spla.eigsh(H, k=pm.grid-2, which='SA', maxiter=1000000)
   energies = solution[0] 
   wavefunctions = solution[1]
  
   # Normalise all wavefunctions
   length = len(wavefunctions[0,:])
   for i in range(0,length):
      wavefunctions[:,i] = wavefunctions[:,i]/(np.linalg.norm(wavefunctions[:,i])*pm.deltax**0.5)

   # Make list of occupied wavefunctions
   occupied = []
   occupied_energies = []
   for i in range(0,pm.NE):
      occupied.append(wavefunctions[:,i])
      occupied_energies.append(energies[i])

   # Make list of empty wavefunctions
   empty = []
   empty_energies = []
   fraction = 0.10 # fraction of empty states to use
   for i in range(pm.NE,int(fraction*(pm.grid-2))):
      empty.append(wavefunctions[:,i])
      empty_energies.append(energies[i])

   # Construct space-time grid
   tau_max = 10.0
   tau_N = 200 # Must be even
   st = SpaceTime(tau_max, tau_N, pm.grid)
  
   # Calculate the non-interacting Green's function in imaginary time
   G0 = non_interacting_greens_function(st, occupied, occupied_energies, empty, empty_energies)

   # Extract the ground-state non-interacting density from G0
   density = []
   for i in range(0,len(st.x1_grid)):
      density.append((-1.0j*G0[(len(st.tau_grid)/2)-1][i][i]).real)

   # Output ground state density
   output_file = open('outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_' + str(pm.NE) + 'gs_mbpt_den.db','w')
   pickle.dump(density,output_file)
   output_file.close()
