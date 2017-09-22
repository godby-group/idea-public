"""Calculates the non-interacting  ground-state electron density and energy 
for a system of N electrons through solving the Schrodinger equation. If the
system is perturbed, the time-dependent electron density and current density 
are calculated. 
"""
from __future__ import division
from __future__ import absolute_import

import pickle
import numpy as np
import scipy as sp
import scipy.sparse as sps
import scipy.linalg as spla
import scipy.sparse.linalg as spsla
from . import RE_cython
from . import results as rs


def construct_K(pm):
    r"""Stores the band elements of the kinetic energy matrix in lower form. 
    The kinetic energy matrix is constructed using a three-point, five-point 
    or seven-point stencil. This yields an NxN band matrix (where N is the 
    number of grid points). For example with N=6 and a three-point stencil:
   
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
        \frac{1}{\delta x^2}
        = [1,-\frac{1}{2}]

    parameters
    ----------
    pm : object
        Parameters object

    returns sparse_matrix
        Kinetic energy matrix
    """
    sd = pm.space.second_derivative_band
    nbnd = len(sd)
    K = np.zeros((nbnd, pm.sys.grid), dtype=np.float)

    for i in range(nbnd):
        K[i,:] = -0.5 * sd[i]

    return K



def construct_V(pm, td):
    r"""Constructs the main diagonal of the potential energy matrix. The 
    potential energy matrix is an NxN diagonal matrix (where N is the number
    of grid points).

    parameters
    ----------
    pm : object
        Parameters object
    td : bool
         - 'False': Construct unperturbed external potential
         - 'True': Construct perturbed external potential

    returns sparse_matrix
        Potential energy matrix
    """
    xgrid = np.linspace(-pm.sys.xmax,pm.sys.xmax,pm.sys.grid)
    V = np.zeros(pm.sys.grid, dtype=np.float)

    if(td == 0):
        V[:] = pm.sys.v_ext(xgrid[:])
    if(td == 1):
        V[:] = (pm.sys.v_ext(xgrid[:]) + pm.sys.v_pert(xgrid[:]))

    return V


def construct_A(pm, H):
    r"""Constructs the matrix A to be used when solving Ax=b in the
    Crank-Nicholson propagation.

    .. math::

        A = I + i \frac{dt}{2} H

    parameters
    ----------
    pm : object
        Parameters object
    H : sparse_matrix
        The Hamiltonian matrix
 
    returns sparse_matrix
        Sparse matrix A
    """
    if(pm.sys.stencil == 3):
        A = 1.0j*(pm.sys.deltat/2.0)*sps.diags([H[1,:], H[0,:], H[1,:]],
        [-1, 0, 1], shape=(pm.sys.grid,pm.sys.grid), format='csc')
    elif(pm.sys.stencil == 5):
        A = 1.0j*(pm.sys.deltat/2.0)*sps.diags([H[2,:], H[1,:], H[0,:], 
            H[1,:], H[2,:]], [-2, -1, 0, 1, 2], shape=(pm.sys.grid,
            pm.sys.grid), format='csc')
    elif(pm.sys.stencil == 7):
        A = 1.0j*(pm.sys.deltat/2.0)*sps.diags([H[3,:], H[2,:], H[1,:], 
            H[0,:], H[1,:], H[2,:], H[3,:]], [-3, -2, -1, 0, 1, 2, 3], 
            shape=(pm.sys.grid,pm.sys.grid), format='csc')

    A += sps.identity(pm.sys.grid) 

    return A


def calculate_current_density(pm, density):
    r"""Calculates the current density of a time evolving wavefunction by 
    solving the continuity equation.

    .. math::

        \frac{\partial n}{\partial t} + \nabla \cdot j = 0

    parameters
    ----------
    pm : object
        Parameters object
    density : array_like
        2D array of the time-dependent density, indexed as       
        density[time_index,space_index]

    returns array_like
        2D array of the current density, indexed as 
        current_density[time_index,space_index]
    """
    pm.sprint('', 1, newline=True)
    current_density = np.zeros((pm.sys.imax,pm.sys.grid), dtype=np.float)
    string = 'NON: calculating current density'
    pm.sprint(string, 1, newline=True)
    for i in range(1, pm.sys.imax):
         string = 'NON: t = {:.5f}'.format(i*pm.sys.deltat)
         pm.sprint(string, 1, newline=False)
         J = np.zeros(pm.sys.grid, dtype=np.float)
         J = RE_cython.continuity_eqn(pm, density[i,:], density[i-1,:])
         current_density[i,:] = J[:]
    pm.sprint('', 1, newline=True)

    return current_density


def main(parameters):
    r"""Calculates the ground-state of the system. If the system is perturbed, 
    the time evolution of the perturbed system is then calculated.

    parameters
    ----------
    parameters : object
        Parameters object

    returns object
        Results object
    """
    pm = parameters
    pm.setup_space()

    # Print to screen
    string = 'NON: constructing arrays'
    pm.sprint(string, 1, newline=True)

    # Construct the kinetic energy matrix
    K = construct_K(pm)
   
    # Construct the potential energy matrix
    V = construct_V(pm, 0)

    # Construct the Hamiltonian matrix
    H = np.copy(K)
    H[0,:] += V[:]

    # Solve the Schrodinger equation
    pm.sprint('NON: computing ground state density',1)
    energies, wavefunctions = spla.eig_banded(H, lower=True)

    # Normalise the wavefunctions
    wavefunctions /= pm.sys.deltax**0.5

    # Calculate the density 
    density = np.sum(wavefunctions[:,:pm.sys.NE]**2, axis=1)

    # Calculate the energy and print to screen
    energy = np.sum(energies[0:pm.sys.NE])
    string = 'NON: ground-state energy = {:.5f}'.format(energy)
    pm.sprint(string, 1, newline=True)

    # Save the ground-state density, energy and external potential
    results = rs.Results()
    results.add(V,'gs_non_vxt')
    results.add(density,'gs_non_den')
    results.add(energy,'gs_non_E')

    # Save the eigenfunctions and eigenvalues
    if pm.non.save_eig:
        results.add(wavefunctions.T,'gs_non_eigf')
        results.add(energies,'gs_non_eigv')

    #Save results
    if (pm.run.save):
        results.save(pm)

    # Propagate through real time
    if(pm.run.time_dependence == True):

        # Print to screen
        string = 'NON: constructing arrays'
        pm.sprint(string, 1, newline=True)

        # Construct the potential energy matrix
        V = construct_V(pm, 1)

        # Construct the Hamiltonian matrix
        H = np.copy(K)
        H[0,:] += V[:]

        # Construct the sparse matrices used in the Crank-Nicholson method
        A = construct_A(pm, H)
        C = 2.0*sps.identity(pm.sys.grid) - A

        # Construct the time-dependent density array 
        density = np.zeros((pm.sys.imax,pm.sys.grid), dtype=np.float)

        # Save the ground-state
        for j in range(pm.sys.NE):
            density[0,:] += abs(wavefunctions[:,j])**2

        # Print to screen
        string = 'NON: real time propagation'
        pm.sprint(string, 1, newline=True)
        
        # Loop over each electron 
        for n in range(pm.sys.NE):

            wavefunction = wavefunctions[:,n].astype(np.cfloat)
  
            # Perform real time iterations
            for i in range(1, pm.sys.imax):

                # Construct the vector b
                b = C*wavefunction   

                # Solve Ax=b
                wavefunction, info = spsla.cg(A,b,x0=wavefunction,
                                 tol=pm.non.rtol_solver)

                # Calculate the density
                density[i,:] += abs(wavefunction[:])**2

                # Calculate the norm of the wavefunction
                normalisation = 1.0 - np.linalg.norm(wavefunction)*pm.sys.deltax**0.5
                string = "NON: N = {}, t = {:.5f}, normalisation error = {:+8.5e}" \
                        .format(n+1, i*pm.sys.deltat, normalisation)
                pm.sprint(string, 1, newline=False)

        # Calculate the current density
        current_density = calculate_current_density(pm, density)

        # Save the time-dependent density, current density and external 
        # potential
        if(pm.run.time_dependence == True):
             results.add(density,'td_non_den')
             results.add(current_density,'td_non_cur')
             results.add(V,'td_non_vxt')     
             if (pm.run.save):
                 results.save(pm)

    return results

