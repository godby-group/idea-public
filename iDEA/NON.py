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
import sprint
import numpy as np
import scipy as sp
import RE_Utilities
import scipy.sparse as sps
import scipy.sparse.linalg as spsla
import results as rs

# Function to construct the kinetic energy K
def constructK():
   K = -0.5*sps.diags([1, -2, 1],[-1, 0, 1], shape=(pm.sys.grid,pm.sys.grid), format='csr')/(pm.sys.deltax**2)
   return K

# Function to construct the potential V
def constructV(td):
   xgrid = np.linspace(-pm.sys.xmax,pm.sys.xmax,pm.sys.grid)
   Vdiagonal = []
   if(td == 0):
      for i in range(0,len(xgrid)):
         Vdiagonal.append(pm.sys.v_ext(xgrid[i]))
   if(td == 1):
      for i in range(0,len(xgrid)):
         Vdiagonal.append(pm.sys.v_ext(xgrid[i]) + pm.sys.v_pert(xgrid[i]))
   V = sps.spdiags(Vdiagonal, 0, pm.sys.grid, pm.sys.grid, format='csr')
   return V

# Function to construct the matrix A from the hamiltonain H
def constructA(H):
   I = sps.identity(pm.sys.grid)
   A = I + 1.0j*(pm.sys.deltat/2.0)*H
   return A

# Function to construct the matrix C from the hamiltonain H
def constructC(H):
   I = sps.identity(pm.sys.grid)
   C = I - 1.0j*(pm.sys.deltat/2.0)*H
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
         string = 'NON: computing time dependent current density t = ' + str(i*pm.sys.deltat)
         sprint.sprint(string,1,pm.run.verbosity,newline=False)
         J = np.zeros(pm.sys.grid)
         J = RE_Utilities.continuity_eqn(pm.sys.grid,pm.sys.deltax,pm.sys.deltat,total_td_density[i+1],total_td_density[i])
         if pm.sys.im==1:
             for j in range(pm.sys.grid):
                 for k in range(j+1):
                     x = k*pm.sys.deltax-pm.sys.xmax
                     J[j] -= abs(pm.sys.im_petrb(x))*total_td_density[i][k]*pm.sys.deltax
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
def main(parameters):
   global pm
   pm = parameters

   # Construct the kinetic energy
   K = constructK()

   # Construct the potential
   V = constructV(0)

   # Constuct the Hamiltonian
   H = K + V

   # Compute first N wavefunctions
   sprint.sprint('NON: computing ground state density',1,pm.run.verbosity)
   solution = spsla.eigs(H, k=pm.sys.NE, which='SR', maxiter=1000000)
   energies = solution[0] 
   wavefunctions = solution[1]

   # Normalise first N wavefunctions
   length = len(wavefunctions[0,:])
   for i in range(0,length):
      wavefunctions[:,i] = wavefunctions[:,i]/(np.linalg.norm(wavefunctions[:,i])*pm.sys.deltax**0.5)

   # Compute first N densities
   densities = []
   for i in range(0,length):
      d = calculateDensity(wavefunctions[:,i])
      densities.append(d)

   # Compute total density
   density = addDensities(densities)

   # Compute ground state energy
   energy = 0.0
   for i in range(0,pm.sys.NE):
      energy += energies[i]
   sprint.sprint('NON: ground state energy: ' + str(energy.real),1,pm.run.verbosity)

   # Save ground state density and energy 
   results = rs.Results()
   results.add(density,'gs_non_den')
   results.add(energy.real,'gs_non_E')

   if(pm.run.save):
      results.save(pm.output_dir + '/raw', pm.run.verbosity)

   # Perform real time iterations
   if(pm.run.time_dependence == True):

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
      for n in range(0,pm.sys.NE):
         wavefunction = wavefunctions[:,n]
         densities = []
         i = 0
         while(i < pm.sys.imax):

            # Construct the vector b
            b = C*wavefunction   

            # Solve Ax=b
            wavefunction, info = spsla.cg(A,b,x0=wavefunction,tol=pm.non.rtol)

            # Calculate the density
            density = calculateDensity(wavefunction)

            # Add current potential and density to output arrays 
            densities.append(density)

            # Calculate the wavefunction normalisation
            normalisation = (np.linalg.norm(wavefunction)*pm.sys.deltax**0.5)
            string = 'NON real time: N = ' + str(n+1) + ', t = ' + str(i*pm.sys.deltat) + ', normalisation = ' + str(normalisation)
      	    sprint.sprint(string,1,pm.run.verbosity,newline=False)

            # iterate
            i = i + 1

         # Add to densities
         td_densities.append(densities)
         sprint.sprint('',1,pm.run.verbosity)
  
      # Calculate total density
      sprint.sprint('NON: computing time dependent density',1,pm.run.verbosity)
      total_density = []
      i = 0
      while(i < pm.sys.imax):
         densities = []
         for n in range(0,pm.sys.NE):
            single_density = td_densities[n]
            densities.append(single_density[i])
         density = addDensities(densities)
         total_density.append(density)
         i = i + 1
      total_density_gs = total_density_gs + total_density

      # Calculate current density
      current_density = calculateCurrentDensity(total_density_gs)
      sprint.sprint('',1,pm.run.verbosity)

      # Save time-dependent density and current
      results.add(np.asarray(total_density),'td_non_den')
      results.add(np.asarray(current_density),'td_non_cur')
 
      if(pm.run.save):
         # no need to save previous results again
         l = ['td_non_den', 'td_non_cur']
         results.save(pm.output_dir + '/raw', pm.run.verbosity, list=l)

   ## Program complete
   #os.system('rm *.pyc')

   return results

