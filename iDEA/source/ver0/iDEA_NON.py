######################################################################################
# Name: Non-interacting approximation                                                #
######################################################################################
# Author(s): Jack Wetherell                                                          #
######################################################################################
# Description:                                                                       #
# Computes non interacting density for given system (n electrons)                    #
#                                                                                    #
#                                                                                    #
######################################################################################
# Notes:                                                                             #
#                                                                                    #
#                                                                                    #
#                                                                                    #
######################################################################################

# Do not run stand-alone
if(__name__ == '__main__'):
    print('do not run stand-alone')
    quit()

# Library imports
import os
import sys
import math
import copy
import cmath
import pickle
import sprint
import numpy as np
import scipy as sp
import RE_Utilities
import parameters as pm
import scipy.sparse as sps
import scipy.sparse.linalg as spla

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

# Function to construct the matrix A from the hamiltonain H
def constructA(H):
   I = sps.identity(pm.grid)
   A = I + 1.0j*(pm.deltat/2.0)*H
   return A

# Function to construct the matrix C from the hamiltonain H
def constructC(H):
   I = sps.identity(pm.grid)
   C = I - 1.0j*(pm.deltat/2.0)*H
   return C

# Function the calculate the density for a given wavefunction
def calculateDensity(wavefunction):
   density = np.zeros(len(wavefunction))
   for i in range(0,len(density)):
      density[i] = abs(wavefunction[i])**2
   return density

# Function to calculate the current density
def calculateCurrentDensity(total_td_density):
    current_density = []
    for i in range(0,len(total_td_density)-1):
         string = 'NON: computing time dependent current density t = ' + str(i*pm.deltat)
         sprint.sprint(string,1,1,pm.msglvl)
         J = np.zeros(pm.jmax)
         J = RE_Utilities.continuity_eqn(pm.jmax,pm.deltax,pm.deltat,total_td_density[i+1],total_td_density[i])
         current_density.append(J)
    return current_density

# Function to combine densities
def addDensities(densities):
   density = [0.0] * int(len(densities[0]))
   for i in range(0,len(densities[0])):
      for d in densities:
         density[i] += d[i]
   return density

# Main function
def main():

   # Construct the kinetic energy
   K = constructK()

   # Construct the potential
   V = constructV(0)

   # Constuct the Hamiltonian
   H = K + V

   # Compute first N wavefunctions
   print 'NON: computing ground state density'
   solution = spla.eigs(H, k=pm.NE, which='SR', maxiter=1000000)
   energies = solution[0] 
   wavefunctions = solution[1]

   # Normalise first N wavefunctions
   length = len(wavefunctions[0,:])
   for i in range(0,length):
      wavefunctions[:,i] = wavefunctions[:,i]/(np.linalg.norm(wavefunctions[:,i])*pm.deltax**0.5)

   # Compute first N densities
   densities = []
   for i in range(0,length):
      d = calculateDensity(wavefunctions[:,i])
      densities.append(d)

   # Compute total density
   density = addDensities(densities)

   # Compute ground state energy
   energy = 0.0
   for i in range(0,pm.NE):
      energy += energies[i]
   print('NON: ground state energy: ' + str(energy.real))

   # Save ground state density to pickle file
   output_file = open('outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_' + str(pm.NE) + 'gs_non_den.db','w')
   pickle.dump(density,output_file)
   output_file.close()

   # Perform real time iterations
   if(pm.TD == 1):

      # Construct evolution matrices
      K = constructK()
      V = constructV(1)
      H = K + V
      A = constructA(H)
      C = constructC(H)

      # Create densities
      td_densities = []
      total_density_gs = []
      total_density_gs.append(density)
      for n in range(0,pm.NE):
         wavefunction = wavefunctions[:,n]
         densities = []
         i = 0
         while(i < pm.imax):

            # Construct the vector b
            b = C*wavefunction   

            # Solve Ax=b
            wavefunction, info = spla.cg(A,b,x0=wavefunction,tol=pm.NON_rtol)

            # Calculate the density
            density = calculateDensity(wavefunction)

            # Add current potential and density to output arrays 
            densities.append(density)

            # Calculate the wavefunction normalisation
            normalisation = (np.linalg.norm(wavefunction)*pm.deltax**0.5)
            string = 'NON real time: N = ' + str(n+1) + ', t = ' + str(i*pm.deltat) + ', normalisation = ' + str(normalisation)
      	    sprint.sprint(string,1,1,pm.msglvl)

            # iterate
            i = i + 1

         # Add to densities
         td_densities.append(densities)
         print
  
      # Calculate total density
      print('NON: computing time dependent density')
      total_density = []
      i = 0
      while(i < pm.imax):
         densities = []
         for n in range(0,pm.NE):
            single_density = td_densities[n]
            densities.append(single_density[i])
         density = addDensities(densities)
         total_density.append(density)
         i = i + 1
      total_density_gs = total_density_gs + total_density

      # Calculate current density
      current_density = calculateCurrentDensity(total_density_gs)
      print

      # Save time dependent data to pickle file (density)
      output_file = open('outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_' + str(pm.NE) + 'td_non_den.db','w')
      pickle.dump(total_density,output_file)
      output_file.close()

      # Save time dependent data to pickle file (current density)
      output_file = open('outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_' + str(pm.NE) + 'td_non_cur.db','w')
      pickle.dump(current_density,output_file)
      output_file.close()

   # Program complete
   os.system('rm *.pyc')

