# Library imports
import sys
import math
import cmath
import numpy as np
import scipy as sp
import scipy.sparse as sps
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sparameters as pm

# Print splash
print('                                                                ')
print('           ****  ****    *    ****     *****       *            ')
print('          *      *   *        *   *    *          * *           ')
print('          *      *   *   *    *    *   *         *   *          ')
print('           ***   * *     *    *     *  *****    *     *         ')
print('              *  *       *    *    *   *       *********        ')
print('              *  *       *    *   *    *      *         *       ')
print('          ****   *       *    ****     ***** *           *      ')
print('                                                                ')
print('  --------------------------------------------------------------')
print('  |                     Single Particle iDEA                   |')
print('  |            (Interacting Dynamic Electrons Approach)        |')      
print('  |                                                            |')
print('  |                   Created by Jack Wetherell                |')
print('  |                    The University of York                  |')
print('  --------------------------------------------------------------')

# Function to construct the potential V
def constructV(time):
   xgrid = np.linspace(-0.5*pm.L,0.5*pm.L,pm.N)
   V = []
   if(time =='i'):
      for i in range(0,len(xgrid)):
         V.append(pm.Vext(xgrid[i]))
   if(time =='r'):
      for i in range(0,len(xgrid)):
         V.append(pm.Vext(xgrid[i]) + pm.Vptrb(xgrid[i]))
   return V

# Function to construct the hamiltonain H
def constructH(time):
   xgrid = np.linspace(-0.5*pm.L,0.5*pm.L,pm.N)
   K = -0.5*sps.diags([1, -2, 1],[-1, 0, 1],shape=(pm.N,pm.N))/(pm.dx**2)
   Vdiagonal = constructV(time)
   V = sps.spdiags(Vdiagonal, 0, pm.N, pm.N, format='csr')
   H = K + V
   return H

# Function to construct the matrix A from the hamiltonain H
def constructA(H,time):
   I = sps.identity(pm.N)
   if(time == 'i'):   
      A = I + 1.0*(pm.cdt/2.0)*H
   if(time =='r'):   
      A = I + 1.0j*(pm.dt/2.0)*H
   return A

# Function to construct the matrix C from the hamiltonain H
def constructC(H,time):
   I = sps.identity(pm.N)
   if(time == 'i'):   
      C = I - 1.0*(pm.cdt/2.0)*H
   if(time == 'r'):   
      C = I - 1.0j*(pm.dt/2.0)*H
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

# Function the return the value of the initial wavefunction at a given x
def initialWavefunction(x):
    return (1.0/math.sqrt(2.0*math.pi))*(math.e**(-0.5*x**2))

# Function to print to screen
def sprint(text):
   sys.stdout.write('\033[K')
   sys.stdout.flush()
   sys.stdout.write('\r' + text)
   sys.stdout.flush()

# Function to save data to file
def save(x,y,filename):
   f = open(filename, 'w')
   for i in range(0,len(x)):
      f.write(str(x[i]) + ' ' + str(y[i]) + '\n')

# Function to animate plots
def animate(potential,densities):
   plt.ion()
   xgrid = np.linspace(-0.5*pm.L,0.5*pm.L,pm.N)
   plt.plot(xgrid,potential)
   plt.ylim(ymax=1.0) 
   plt.ylim(ymin=-0.1)
   line, = plt.plot(xgrid,densities[0])
   for i in range(1,len(densities)):
      sprint('plotting real time density, timestep = ' + str(i))
      line.set_ydata(densities[i])
      plt.draw()  

# Main function
def main():

   # Create the grid
   xgrid = np.linspace(-0.5*pm.L,0.5*pm.L,pm.N)

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
   while(i < pm.cI and convergence > pm.ctol):

      # Construct the vector b
      b = C*wavefunction   
      
      # Set the old wavefunction
      old_wavefunction = wavefunction

      # Solve Ax=b
      wavefunction = spla.spsolve(A,b)

      # Normalise the wavefunction
      wavefunction = wavefunction/(np.linalg.norm(wavefunction)*pm.dx**0.5)

      # Calculate the density
      density = calculateDensity(wavefunction)

      # Calculate the wavefunction convergence
      convergence = np.linalg.norm(wavefunction-old_wavefunction)
      sprint('complex time = ' + str(i*pm.cdt) + ', convergence = ' + str(convergence))
      
      # iterate
      i = i + 1

   # Calculate the groundstate energy
   energy = calculateEnergy(H,wavefunction)
   print
   print 'ground state energy =', energy

   # Save data to file
   if(pm.saveGround == 1):
      save(xgrid,V,'Potential.dat')
      save(xgrid,density,'GroundState.dat')

   # Construct the potential
   V = constructV('r')

   # Construct matrices
   H = constructH('r')
   A = constructA(H,'r')
   C = constructC(H,'r')

   # Perform real time iterations until completed
   i = 0
   if(pm.TD != 1):
      i = pm.rI
   densities = []
   while(i < pm.rI):

      # Construct the vector b
      b = C*wavefunction   

      # Solve Ax=b
      wavefunction = spla.spsolve(A,b)

      # Calculate the density
      density = calculateDensity(wavefunction)
      densities.append(density)

      # Calculate the wavefunction normalisation
      normalisation = (np.linalg.norm(wavefunction)*pm.dx**0.5)
      sprint('real time = ' + str(i*pm.dt) + ', normalisation = ' + str(normalisation))

      # Save data to file if required
      if(pm.saveGround == 1 and pm.saveTime == i):
         save(xgrid,V,'PtrbPotential.dat')
         save(xgrid,density,'Density(i=' + str(i) + ').dat')

      # iterate
      i = i + 1

   # Animation
   if(pm.animatePlot == 1):
      print
      animate(V,densities)
      print
 
   # Program Complete
   print('All jobs done.')

# Run stand-alone
if(__name__ == '__main__'):
   main()
