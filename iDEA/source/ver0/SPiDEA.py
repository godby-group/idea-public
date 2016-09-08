######################################################################################
# Name: 1 electron Many Body. Single Particle iDEA (SPiDEA)                          #
######################################################################################
# Author(s): Jack Wetherell                                                          #
######################################################################################
# Description:                                                                       #
# Computes exact many body wavefunction and density for one electron systems         #
#                                                                                    #
#                                                                                    #
######################################################################################
# Notes:                                                                             #
# Use this to aid understanding of MB calculations                                   #
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
import numpy as np
import scipy as sp
import RE_Utilities
import parameters as pm
import scipy.sparse as sps
import scipy.sparse.linalg as spla

# Function to construct the potential V
def constructV(time):
   xgrid = np.linspace(-pm.xmax,pm.xmax,pm.grid)
   V = []
   if(time =='i'):
      for i in range(0,len(xgrid)):
         V.append(pm.well(xgrid[i]))
   if(time =='r'):
      for i in range(0,len(xgrid)):
         V.append(pm.well(xgrid[i]) + pm.petrb(xgrid[i]))
   return V

# Function to construct the hamiltonain H
def constructH(time):
   xgrid = np.linspace(-pm.xmax,pm.xmax,pm.grid)
   K = -0.5*sps.diags([1, -2, 1],[-1, 0, 1],shape=(pm.grid,pm.grid))/(pm.deltax**2)
   Vdiagonal = constructV(time)
   V = sps.spdiags(Vdiagonal, 0, pm.grid, pm.grid, format='csr')
   H = K + V
   return H

# Function to construct the matrix A from the hamiltonain H
def constructA(H,time):
   I = sps.identity(pm.grid)
   if(time == 'i'):   
      A = I + 1.0*(pm.cdeltat/2.0)*H
   if(time =='r'):   
      A = I + 1.0j*(pm.deltat/2.0)*H
   return A

# Function to construct the matrix C from the hamiltonain H
def constructC(H,time):
   I = sps.identity(pm.grid)
   if(time == 'i'):   
      C = I - 1.0*(pm.cdeltat/2.0)*H
   if(time == 'r'):   
      C = I - 1.0j*(pm.deltat/2.0)*H
   return C

# Function to return the energy of a wavefunction given the hamiltonain H
def calculateEnergy(H,wavefunction):
   A = H*wavefunction
   B = wavefunction
   energies = []
   for i in range(0,len(A)):
      energies.append(A[i]/B[i])
   energy = np.average(np.array(energies))
   return energy

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
         string = 'MB: computing time dependent current density t = ' + str(i*pm.deltat)
         sprint(string)
         J = np.zeros(pm.jmax)
         J = RE_Utilities.continuity_eqn(pm.jmax,pm.deltax,pm.deltat,total_td_density[i+1],total_td_density[i])
         if pm.im == 1:
             for j in range(pm.jmax):
                 for k in range(j+1):
                     x = k*pm.deltax-pm.xmax
                     J[j] -= abs(pm.im_petrb(x))*total_td_density[i][k]*pm.deltax
         current_density.append(J)
    return current_density

# Function the return the value of the initial wavefunction at a given x
def initialWavefunction(x):
    return (1.0/math.sqrt(2.0*math.pi))*(math.e**(-0.5*x**2))

# Function to print to screen replacing the current line
def sprint(text):
   sys.stdout.write('\033[K')
   sys.stdout.flush()
   sys.stdout.write('\r' + text)
   sys.stdout.flush()

# Main function
def main():

   # Create the grid
   xgrid = np.linspace(-0.5*pm.xmax,0.5*pm.xmax,pm.grid)

   # Construct the potential
   V = constructV('i')

   # Construct matrices
   H = constructH('i')
   A = constructA(H,'i')
   C = constructC(H,'i')

   # Construct initial wavefunction
   wavefunction = initialWavefunction(xgrid)
   
   # Perform complex time iterations until converged
   i = 0
   convergence = 1.0
   cI = int(pm.ctmax/pm.cdeltat)
   while(i < cI and convergence > pm.ctol):

      # Construct the vector b
      b = C*wavefunction   
      
      # Set the old wavefunction
      old_wavefunction = wavefunction

      # Solve Ax=b
      wavefunction, info = spla.cg(A,b,x0=wavefunction,tol=pm.ctol)

      # Normalise the wavefunction
      wavefunction = wavefunction/(np.linalg.norm(wavefunction)*pm.deltax**0.5)

      # Calculate the density
      density = calculateDensity(wavefunction)

      # Calculate the wavefunction convergence
      convergence = np.linalg.norm(wavefunction-old_wavefunction)
      sprint('many body complex time: t = ' + str(i*pm.cdeltat) + ', convergence = ' + str(convergence))
      
      # iterate
      i = i + 1

   # Calculate the groundstate energy
   energy = calculateEnergy(H,wavefunction)
   print
   print 'many body complex time: ground state energy =', energy

   # Save ground state energy to file
   output_file = open('outputs/' + str(pm.run_name) + '/data/' + str(pm.run_name) + '_1gs_ext_E.dat','w')
   output_file.write(str(energy))
   output_file.close()

   # Save ground state data to pickle file
   output_file = open('outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_1gs_ext_den.db','w')
   pickle.dump(density,output_file)
   output_file.close()
   output_file = open('outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_1gs_ext_vxt.db','w')
   pickle.dump(V,output_file)
   output_file.close()	

   # Construct the potential
   V = constructV('r')

   # Construct matrices
   H = constructH('r')
   A = constructA(H,'r')
   C = constructC(H,'r')

   # Perform real time iterations until completed
   i = 0
   if(pm.TD != 1):
      i = pm.imax
   densities = []
   potentials = []
   while(i < pm.imax):

      # Construct the vector b
      b = C*wavefunction   

      # Solve Ax=b
      wavefunction, info = spla.bicgstab(A,b,x0=wavefunction,tol=pm.ctol)

      # Calculate the density
      density = calculateDensity(wavefunction)

      # Add current potential and density to output arrays 
      # Note: This appears inefficient but will allow time dependent potentials to be more easily added
      densities.append(density)
      Vout = copy.copy(V)
      for v in Vout:
         v = v.real
      potentials.append(Vout)

      # Calculate the wavefunction normalisation
      normalisation = (np.linalg.norm(wavefunction)*pm.deltax**0.5)
      sprint('many body real time: t = ' + str(i*pm.deltat) + ', normalisation = ' + str(normalisation))

      # iterate
      i = i + 1

   # Save time dependant data to pickle file
   if(pm.TD == 1):
      current_density = calculateCurrentDensity(densities)
      output_file = open('outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_1td_ext_den.db','w')
      pickle.dump(densities,output_file)
      output_file.close()
      output_file = open('outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_1td_ext_cur.db','w')
      pickle.dump(current_density,output_file)
      output_file.close()
      output_file = open('outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_1td_ext_vxt.db','w')
      pickle.dump(potentials,output_file)
      output_file.close()

   # Program Complete
   if(pm.TD == 1):
      print
   os.system('rm *.pyc')
