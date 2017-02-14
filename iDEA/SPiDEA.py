"""Computes charge density for given one-electron system. For ground state calculations the code outputs the ground state charge density
and energy of the system and the system's external potential. For time dependent calculation the code also outputs the time-dependent charge and current
densities.
"""


import os
import sys
import math
import copy
import pickle
import numpy as np
import scipy as sp
import RE_Utilities
import scipy.sparse as sps
import scipy.sparse.linalg as spsla
import results as rs


def constructK():
   r"""Constructs the kinetic energy operator K on the system grid
    
   This is constructed using a second-order stencil yielding a tri-diagonal NxN matrix (where 
   N is the number of grid points). For example with N=6:
   
   .. math::

       K = -\frac{1}{2} \frac{d^2}{dx^2}=
       -\frac{1}{2} \begin{pmatrix}
       -2 & 1 & 0 & 0 & 0 & 0 \\
       1 & -2 & 1 & 0 & 0 & 0 \\
       0 & 1 & -2 & 1 & 0 & 0 \\
       0 & 0 & 1 & -2 & 1 & 0 \\
       0 & 0 & 0 & 1 & -2 & 1 \\
       0 & 0 & 0 & 0 & 1 & -2 
       \end{pmatrix}
       \frac{1}{\Delta x^2}

   parameters
   ----------

   returns sparse_matrix
   """
   K = -0.5*sps.diags([1, -2, 1],[-1, 0, 1], shape=(pm.sys.grid,pm.sys.grid), format='csr')/(pm.sys.deltax**2)
   return K


def constructV(td):
   r"""Constructs the potential energy operator V on the system grid
   
   V will contain V(x) sampled on the system grid along the diagonal yielding a NxN diagonal matrix (where 
   N is the number of grid points).

   parameters
   ----------
   td : bool
        - 'False': Construct external potential
        - 'True': Construct peturbed potential

   returns sparse_matrix
   """
   xgrid = np.linspace(-pm.sys.xmax,pm.sys.xmax,pm.sys.grid)
   Vdiagonal = np.empty(pm.sys.grid)
   if(td == 0):
      Vdiagonal[:] = pm.sys.v_ext(xgrid[:])
   if(td == 1):
      Vdiagonal[:] = (pm.sys.v_ext(xgrid[:]) + pm.sys.v_pert(xgrid[:]))
   V = sps.spdiags(Vdiagonal, 0, pm.sys.grid, pm.sys.grid, format='csr')
   return V


def constructA(H,td):
   r"""Constructs the matrix A to be used in the crank-nicholson solution of Ax=b when evolving the wavefunction in time

   .. math::

       A = I + i \frac{dt}{2} H

   parameters
   ----------
   H : sparse_matrix
        The Hamiltonian matrix
   td : bool
        - 'False': Construct for imaginary-time propagation
        - 'True': Construct for real-time propagation

   returns sparse_matrix
   """
   I = sps.identity(pm.sys.grid)
   if(td == 0):   
      A = I + 1.0*(pm.ext.cdeltat/2.0)*H
   if(td == 1):   
      A = I + 1.0j*(pm.sys.deltat/2.0)*H
   return A


def constructC(H,td):
   r"""Constructs the matrix C to be used in the crank-nicholson solution of Ax=b when evolving the wavefunction in time

   .. math::

       C = I - i \frac{dt}{2} H

   parameters
   ----------
   H : sparse_matrix
        The Hamiltonian matrix
   td : bool
        - 'False': Construct for imaginary-time propagation
        - 'True': Construct for real-time propagation

   returns sparse_matrix
   """
   I = sps.identity(pm.sys.grid)
   if(td == 0):   
      C = I - 1.0*(pm.ext.cdeltat/2.0)*H
   if(td == 1):   
      C = I - 1.0j*(pm.sys.deltat/2.0)*H
   return C


def calculateEnergy(H,wavefunction):
   r"""Calculates the energy of a given single particle wavefunction by ensuring

   .. math::

       H \psi = E \psi

   parameters
   ----------
   H : sparse_matrix
        The Hamiltonian matrix
   wavefunction : array_like
        Single particle wavefunction 

   returns double
      Energy
   """
   A = H*wavefunction
   B = wavefunction
   energies = []
   for i in range(0,len(A)):
      energies.append(A[i]/B[i])
   energy = np.average(np.array(energies))
   return energy


def calculateDensity(wavefunction):
   r"""Calculates the electron density from a given wavefunction

   .. math::

       n \left(x \right) = |\psi \left( x\right)|^2

   parameters
   ----------
   wavefunction : array_like
        The wavefunction

   returns array_like
   """
   density = np.empty(pm.sys.grid)
   density[:] = abs(wavefunction[:])**2
   return density


def calculateCurrentDensity(total_td_density):
   r"""Calculates the current density of a time evolving wavefunction by solving the continuity equation.

   .. math::

       \frac{\partial n}{\partial t} + \nabla \cdot j = 0
       
   Note: This function requires RE_Utilities.so

   parameters
   ----------
   total_td_density : array_like
      Time dependent density of the system indexed as total_td_density[time_index][space_index]

   returns array_like
      Time dependent current density indexed as current_density[time_index][space_index]
   """
   current_density = []
   for i in range(0,len(total_td_density)-1):
      string = 'NON: computing time dependent current density t = ' + str(i*pm.sys.deltat)
      pm.sprint(string,1,newline=False)
      J = np.zeros(pm.sys.grid)
      J = RE_Utilities.continuity_eqn(pm.sys.grid,pm.sys.deltax,pm.sys.deltat,total_td_density[i+1],total_td_density[i])
      if pm.sys.im==1: # Here we take account of the decay of the density due to the imaginary boundary consitions (if present)
         for j in range(pm.sys.grid):
            for k in range(j+1):
               x = k*pm.sys.deltax-pm.sys.xmax
               J[j] -= abs(pm.sys.im_petrb(x))*total_td_density[i][k]*pm.sys.deltax
      current_density.append(J)
   return np.asarray(current_density)


def initialWavefunction(x):
   r"""Calculates the value of the initial wavefunction at a given x


   parameters
   ----------
   x : float
      given x

   returns array_like
      Initial guess for wavefunction
   """
   return (1.0/math.sqrt(2.0*math.pi))*(math.e**(-0.5*x**2))


def main(parameters):
   r"""Performs calculation of the one-electron density

   parameters
   ----------
   parameters : object
      Parameters object

   returns object
      Results object
   """
   global pm
   pm = parameters

   # Create the grid
   xgrid = np.linspace(-pm.sys.xmax,pm.sys.xmax,pm.sys.grid)

   # Construct the kinetic energy
   K = constructK()
   
   # Construct the potential
   V = constructV(0)

   # Construct matrices
   H = K + V
   A = constructA(H,False)
   C = constructC(H,False)

   # Construct initial wavefunction
   wavefunction = initialWavefunction(xgrid)
   
   # Perform complex time iterations until converged
   i = 0
   convergence = 1.0
   cI = int(pm.ext.ctmax/pm.ext.cdeltat)
   while(i < cI and convergence > pm.ext.ctol):

      # Construct the vector b
      b = C*wavefunction   
      
      # Set the old wavefunction
      old_wavefunction = wavefunction

      # Solve Ax=b
      wavefunction, info = spsla.cg(A,b,x0=wavefunction,tol=pm.ext.ctol)

      # Normalise the wavefunction
      wavefunction = wavefunction/(np.linalg.norm(wavefunction)*pm.sys.deltax**0.5)

      # Calculate the density
      density = calculateDensity(wavefunction)

      # Calculate the wavefunction convergence
      convergence = np.linalg.norm(wavefunction-old_wavefunction)
      string = 'EXT: complex time, t = ' + str(i*pm.ext.cdeltat) + ', convergence = ' + str(convergence)
      pm.sprint(string,1,newline=False)
      
      # iterate
      i = i + 1

   # Calculate the groundstate energy
   energy = calculateEnergy(H,wavefunction)
   print
   print 'EXT: complex time, ground state energy =', energy

   # Save ground state energy
   results = rs.Results()
   results.add(energy, 'gs_ext_E')

   # Save ground state density and external potential
   results.add(density, 'gs_ext_den')
   results.add(V, 'gs_ext_vxt')
   if pm.run.save:
      results.save(pm)

   # Construct the potential
   V = constructV(1)

   # Construct matrices
   H = K + V
   A = constructA(H,True)
   C = constructC(H,True)

   # Perform real time iterations until completed
   i = 0
   if(pm.run.time_dependence != True):
      i = pm.sys.imax
   densities = []
   potentials = []
   while(i < pm.sys.imax):

      # Construct the vector b
      b = C*wavefunction   

      # Solve Ax=b
      wavefunction, info = spsla.bicgstab(A,b,x0=wavefunction,tol=pm.ext.ctol)

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
      normalisation = (np.linalg.norm(wavefunction)*pm.sys.deltax**0.5)
      string = 'EXT: real time, t = ' + str(i*pm.sys.deltat) + ', normalisation = ' + str(normalisation)
      pm.sprint(string,1,newline=False)
      
      # iterate
      i = i + 1

   # Save time dependant data to pickle file
   if(pm.run.time_dependence == True):
      current_density = calculateCurrentDensity(densities)
      print
   
      results.add(densities,'td_ext_den')
      results.add(current_density,'td_ext_cur')
      results.add(potentials,'td_ext_vxt')
      
      if pm.run.save:
         l = ['td_ext_den','td_ext_cur','td_ext_vxt']
         results.save(pm, list=l)
      

   # Program Complete
   os.system('rm *.pyc')

   return results

