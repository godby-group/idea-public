"""Computes non-interacting charge density for given system. For ground state calculations the code outputs the non-interacting orbitals 
and energies of the system and the ground-state charge density. For time dependent calculation the code also outputs the time-dependent charge and current
densities.
"""


import os
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


def constructA(H):
   r"""Constructs the matrix A to be used in the crank-nicholson solution of Ax=b when evolving the wavefunction in time

   .. math::

       A = I + i \frac{dt}{2} H

   parameters
   ----------
   H: sparse_matrix
        The Hamiltonian matrix

   returns sparse_matrix
   """
   I = sps.identity(pm.sys.grid)
   A = I + 1.0j*(pm.sys.deltat/2.0)*H
   return A


def constructC(H):
   r"""Constructs the matrix C to be used in the crank-nicholson solution of Ax_n=b (where b = C*x_(n-1) where x_(n-1) is 
   the wavefunction from the last timestep) when evolving the wavefunction in time

    .. math::

       C = I - i \frac{dt}{2} H

   parameters
   ----------
   H : sparse_matrix
        The Hamiltonian matrix

   returns sparse_matrix
   """
   I = sps.identity(pm.sys.grid)
   C = I - 1.0j*(pm.sys.deltat/2.0)*H
   return C


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


def addDensities(densities):
   r"""Adds together all of the occupied densities to give the total system density

   parameters
   ----------
   densities : array_like
      Array of densities to be added indexed as densities[electron_index][space_index]

   returns array_like
      Total system density
   """
   density = np.zeros(pm.sys.grid)
   for i in range(pm.sys.NE):
      density[:] += densities[i][:]
   return density


def main(parameters):
   r"""Performs calculation of the non-interacting density

   parameters
   ----------
   parameters : object
      Parameters object

   returns object
      Results object
   """
   global pm
   pm = parameters

   # Construct the kinetic energy
   K = constructK()

   # Construct the potential
   V = constructV(0)

   # Constuct the Hamiltonian
   H = K + V

   # Compute wavefunctions
   pm.sprint('NON: computing ground state density',1)
   energies, wavefunctions = spsla.eigs(H, k=pm.sys.grid-2, which='SR', maxiter=1000000)
   
   # Order by energy
   indices = np.argsort(energies)
   energies = energies[indices]
   wavefunctions = ((wavefunctions.T)[indices]).T 

   # Normalise wavefunctions
   wavefunctions /= pm.sys.deltax**0.5

   # Compute first N densities
   densities = np.empty((pm.sys.NE,pm.sys.grid))
   for i in range(0,pm.sys.NE):
      densities[i,:] = calculateDensity(wavefunctions[:,i])

   # Compute total density
   density = addDensities(densities)

   # Compute ground state energy
   energy = 0.0
   for i in range(0,pm.sys.NE):
      energy += energies[i]
   pm.sprint('NON: ground state energy: ' + str(energy.real),1)

   # Save ground state density and energy 
   results = rs.Results()
   results.add(density,'gs_non_den')
   results.add(energy.real,'gs_non_E')

   if pm.non.save_eig:
       results.add(wavefunctions.T,'gs_non_eigf')
       results.add(energies,'gs_non_eigv')

   if(pm.run.save):
      results.save(pm)

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
      	    pm.sprint(string,1,newline=False)

            # iterate
            i = i + 1

         # Add to densities
         td_densities.append(densities)
         pm.sprint('',1)
  
      # Calculate total density
      pm.sprint('NON: computing time dependent density',1)
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
      pm.sprint('',1)

      # Save time-dependent density and current
      results.add(np.asarray(total_density),'td_non_den')
      results.add(np.asarray(current_density),'td_non_cur')
 
      if(pm.run.save):
         # no need to save previous results again
         l = ['td_non_den', 'td_non_cur']
         results.save(pm, list=l)

   return results

