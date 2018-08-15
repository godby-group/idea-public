"""Calculates the exact ground-state electron density and energy for a system of one electron through
solving the Schroedinger equation. If the system is perturbed, the time-dependent electron density and 
current density are calculated. Excited states of the unperturbed system can also be calculated.
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import numpy.linalg as npla
import scipy as sp
import scipy.sparse as sps
import scipy.linalg as spla
import scipy.sparse.linalg as spsla

from . import EXT_cython
from . import results as rs


def construct_K(pm):
    r"""Stores the band elements of the kinetic energy matrix in lower form. The kinetic energy matrix 
    is constructed using a three-point, five-point or seven-point stencil. This yields an NxN band 
    matrix (where N is the number of grid points). For example with N=6 and a three-point stencil:
   
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
        = [\frac{1}{\delta x^2},-\frac{1}{2 \delta x^2}]  

    parameters
    ----------
    pm : object
        Parameters object

    returns array_like
        2D array containing the band elements of the kinetic energy matrix, indexed as 
        K[band,space_index]
    """
    # Stencil to use
    sd = pm.space.second_derivative_band
    nbnd = len(sd)

    # Band elements
    K = np.zeros((nbnd, pm.space.npt), dtype=np.float)
    for i in range(nbnd):
        K[i,:] = -0.5 * sd[i]

    return K


def construct_A(pm, H):
    r"""Constructs the sparse matrix A, used when solving Ax=b in the Crank-Nicholson propagation.

    .. math::

        A = I + i \frac{\delta t}{2} H

    parameters
    ----------
    pm : object
        Parameters object
    H : array_like
        2D array containing the band elements of the Hamiltonian matrix, indexed as 
        H[band,space_index]
 
    returns sparse_matrix
        The sparse matrix A
    """
    # Construct the sparse matrix from the band elements of the Hamiltonian matrix
    prefactor = 0.5j*pm.sys.deltat
    if(pm.sys.stencil == 3):
        A = prefactor*sps.diags([H[1,:],H[0,:],H[1,:]], [-1, 0, 1], shape=(pm.space.npt,pm.space.npt), format='csc', 
            dtype=np.cfloat)
    elif(pm.sys.stencil == 5):
        A = prefactor*sps.diags([H[2,:],H[1,:],H[0,:],H[1,:],H[2,:]], [-2, -1, 0, 1, 2], 
            shape=(pm.space.npt,pm.space.npt), format='csc', dtype=np.cfloat)
    elif(pm.sys.stencil == 7):
        A = prefactor*sps.diags([H[3,:],H[2,:],H[1,:],H[0,:],H[1,:],H[2,:],H[3,:]], [-3, -2, -1, 0, 1, 2, 3], 
            shape=(pm.space.npt,pm.space.npt), format='csc', dtype=np.cfloat)

    # Add the identity matrix
    A += sps.identity(pm.space.npt, dtype=np.cfloat)  

    return A

def calculate_current_density(pm, density):
    r"""Calculates the current density from the time-dependent electron density by solving the 
    continuity equation.

    .. math:: 

        \frac{\partial n}{\partial t} + \frac{\partial j}{\partial x} = 0

    parameters
    ----------
    pm : object
        Parameters object
    density : array_like
        2D array of the time-dependent density, indexed as density[time_index,space_index]

    returns array_like
        2D array of the current density, indexed as current_density[time_index,space_index]
    """
    pm.sprint('', 1)
    string = 'EXT: calculating current density'
    pm.sprint(string, 1)
    current_density = np.zeros((pm.sys.imax,pm.space.npt), dtype=np.float)
    for i in range(1, pm.sys.imax):
         string = 'EXT: t = {:.5f}'.format(i*pm.sys.deltat)
         pm.sprint(string, 1, newline=False)
         J = np.zeros(pm.space.npt, dtype=np.float)
         J = EXT_cython.continuity_eqn(pm, density[i,:], density[i-1,:])
         current_density[i,:] = J[:]
    pm.sprint('', 1)

    return current_density


def main(parameters):
    r"""Calculates the ground-state of the system. If the system is perturbed, the time evolution of 
    the perturbed system is then calculated.

    parameters
    ----------
    parameters : object
        Parameters object

    returns object
        Results object
    """
    # Array initialisations
    pm = parameters
    string = 'EXT: constructing arrays'
    pm.sprint(string, 1)
    pm.setup_space()

    # Construct the kinetic energy matrix
    K = construct_K(pm)

    # Construct the Hamiltonian matrix
    H = np.copy(K)
    H[0,:] += pm.space.v_ext[:]

    # Solve the Schroedinger equation
    string = 'EXT: calculating the ground-state density'
    pm.sprint(string, 1)
    energies, wavefunctions = spla.eig_banded(H, lower=True)

    # Normalise the wavefunctions
    wavefunctions /= np.sqrt(pm.space.delta)
 
    # Calculate the ground-state density 
    density = np.absolute(wavefunctions[:,0])**2

    # Calculate the ground-state energy
    energy = energies[0]
    string = 'EXT: ground-state energy = {:.5f}'.format(energy)
    pm.sprint(string, 1)

    # Save the quantities to file
    results = rs.Results()
    results.add(pm.space.v_ext,'gs_ext_vxt')
    results.add(density,'gs_ext_den')
    results.add(energy,'gs_ext_E')
    results.add(wavefunctions.T,'gs_ext_eigf')
    results.add(energies,'gs_ext_eigv')
    if (pm.run.save):
        results.save(pm)

    # Propagate through real time
    if(pm.run.time_dependence):

        # Print to screen
        string = 'EXT: constructing arrays'
        pm.sprint(string, 1, newline=True)

        # Construct the Hamiltonian matrix
        H = np.copy(K)
        H[0,:] += pm.space.v_ext[:]
        if(pm.sys.im == 1):
            H = H.astype(np.cfloat)
        H[0,:] += pm.space.v_pert[:]

        # Construct the sparse matrices used in the Crank-Nicholson method
        A = construct_A(pm, H)
        C = 2.0*sps.identity(pm.space.npt, dtype=np.cfloat) - A

        # Construct the time-dependent density array 
        density = np.zeros((pm.sys.imax,pm.space.npt), dtype=np.float)

        # Save the ground-state
        wavefunction = wavefunctions[:,0].astype(np.cfloat)
        density[0,:] = np.absolute(wavefunction[:])**2
    
        # Print to screen
        string = 'EXT: real time propagation'
        pm.sprint(string, 1)
        
        # Perform real time iterations
        for i in range(1, pm.sys.imax):

            # Construct the vector b
            b = C*wavefunction   

            # Solve Ax=b
            wavefunction, info = spsla.cg(A, b, x0=wavefunction, tol=pm.ext.rtol_solver)

            # Normalise the wavefunction 
            norm = npla.norm(wavefunction)*np.sqrt(pm.space.delta)
            wavefunction /= norm
            string = 'EXT: t = {:.5f}, normalisation = {}'.format(i*pm.sys.deltat, norm**2)
            pm.sprint(string, 1, newline=False)

            # Calculate the density
            density[i,:] = np.absolute(wavefunction[:])**2

        # Calculate the current density
        current_density = calculate_current_density(pm, density)

        # Save the quantities to file
        results.add(density,'td_ext_den')
        results.add(current_density,'td_ext_cur')
        results.add(pm.space.v_ext+pm.space.v_pert,'td_ext_vxt')     
        if(pm.run.save):
            results.save(pm)

    return results
