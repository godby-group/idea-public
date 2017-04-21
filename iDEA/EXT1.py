"""Calculates the exact ground-state electron density and energy for a system 
of one electron through solving the Schrodinger equation. If the system is 
perturbed, the time-dependent electron density and current density are 
calculated. Excited states of the unperturbed system can also be calculated.
"""


import copy
import pickle
import numpy as np
import scipy as sp
import RE_Utilities
import scipy.sparse as sps
import scipy.linalg as spla
import scipy.sparse.linalg as spsla
import results as rs


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
    if(pm.sys.grid < pm.sys.stencil):
        raise ValueError("Insufficient spatial grid points.")
    if(pm.sys.stencil == 3):
        K = np.zeros((2,pm.sys.grid), dtype=np.float)
        K[0,:] = np.ones(pm.sys.grid)/(pm.sys.deltax**2) 							
        K[1,:] = -0.5*np.ones(pm.sys.grid)/(pm.sys.deltax**2) 
    elif(pm.sys.stencil == 5):
        K = np.zeros((3,pm.sys.grid), dtype=np.float)
        K[0,:] = (5.0/4.0)*np.ones(pm.sys.grid)/(pm.sys.deltax**2) 							
        K[1,:] = -(2.0/3.0)*np.ones(pm.sys.grid)/(pm.sys.deltax**2) 
        K[2,:] = (1.0/24.0)*np.ones(pm.sys.grid)/(pm.sys.deltax**2) 
    elif(pm.sys.stencil == 7):
        K = np.zeros((4,pm.sys.grid), dtype=np.float)
        K[0,:] = (49.0/36.0)*np.ones(pm.sys.grid)/(pm.sys.deltax**2) 							
        K[1,:] = -(3.0/4.0)*np.ones(pm.sys.grid)/(pm.sys.deltax**2) 
        K[2,:] = (3.0/40.0)*np.ones(pm.sys.grid)/(pm.sys.deltax**2) 
        K[3,:] = -(1.0/180.0)*np.ones(pm.sys.grid)/(pm.sys.deltax**2) 
    else:
        raise ValueError("pm.sys.stencil must be either 3, 5 or 7.")    

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

    I = sps.identity(pm.sys.grid)
    A += I

    return A


def calculate_current_density(pm, density):
    r"""Calculates the current density of a time evolving wavefunction by 
    solving the continuity equation.

    .. math::

        \frac{\partial n}{\partial t} + \nabla \cdot j = 0
       
    Note: This function requires RE_Utilities.so

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
    pm.sprint('',1,newline=True)
    current_density = np.zeros((pm.sys.imax,pm.sys.grid), dtype=np.float)
    string = 'EXT: calculating current density'
    pm.sprint(string,1,newline=True)
    for i in range(pm.sys.imax):
         string = 'EXT: t = {:.5f}'.format((i+1)*pm.sys.deltat)
         pm.sprint(string,1,newline=False)
         J = np.zeros(pm.sys.grid)
         J = RE_Utilities.continuity_eqn(pm.sys.grid, pm.sys.deltax,
             pm.sys.deltat, density[i+1,:], density[i,:])
         if(pm.sys.im == 1):
             for j in range(pm.sys.grid):
                 for k in range(j+1):
                     x = k*pm.sys.deltax-pm.sys.xmax
                     J[j] -= abs(pm.sys.im_petrb(x))*density[i,k]*(
                             pm.sys.deltax)
         current_density[i,:] = J[:]
    pm.sprint('',1,newline=True)

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

    # Print to screen
    string = 'EXT: constructing arrays'
    pm.sprint(string, 1, newline=True)

    # Construct the kinetic energy matrix
    K = construct_K(pm)
   
    # Construct the potential energy matrix
    V = construct_V(pm, 0)

    # Construct the Hamiltonian matrix
    H = copy.copy(K)
    H[0,:] += V[:]

    # Solve the Schrodinger equation
    energies, wavefunctions = spla.eig_banded(H, lower=True, select='i', 
                              select_range=(0,pm.ext.excited_states))

    # Normalise the wavefunctions and calculate the densities 
    densities = np.zeros((pm.ext.excited_states+1,pm.sys.grid), dtype=np.float)
    for j in range(pm.ext.excited_states+1):
        normalisation = (np.linalg.norm(wavefunctions[:,j])*pm.sys.deltax**0.5)
        wavefunctions[:,j] /= normalisation
        densities[j,:] = abs(wavefunctions[:,j])**2

    # Print energies to screen
    string = 'EXT: ground-state energy = {:.5f}'.format(energies[0])
    pm.sprint(string, 1, newline=True)
    if(pm.ext.excited_states != 0):
        for j in range(pm.ext.excited_states):
            string = 'EXT: {0} excited-state energy = {1:.5f}'.format(j+1, energies[j+1])
            pm.sprint(string, 1, newline=True)

    # Save the ground-state density, energy and external potential
    results = rs.Results()
    results.add(V,'gs_ext_vxt')
    results.add(densities[0,:],'gs_ext_den')
    results.add(energies[0],'gs_ext_E')

    # Save excited-state densities and energies if necessary
    if(pm.ext.excited_states != 0):
        for j in range(pm.ext.excited_states):
            results.add(densities[j+1,:],'gs_ext_den{}'.format(j+1))
            results.add(energies[j+1],'gs_ext_E{}'.format(j+1))
    if (pm.run.save):
        results.save(pm)

    # Propagate through real time
    if(pm.run.time_dependence == True):

        # Print to screen
        string = 'EXT: constructing arrays'
        pm.sprint(string, 1, newline=True)

        # Construct the potential energy matrix
        V = construct_V(pm, 1)

        # Construct the Hamiltonian matrix
        H = copy.copy(K)
        H[0,:] += V[:]

        # Construct the sparse matrices used in the Crank-Nicholson method
        A = construct_A(pm, H)
        C = 2.0*sps.identity(pm.sys.grid) - A

        # Construct the time-dependent density array 
        density = np.zeros((pm.sys.imax+1,pm.sys.grid), dtype=np.float)

        # Save the ground-state
        wavefunction = wavefunctions[:,0].astype(np.cfloat)
        density[0,:] = abs(wavefunction[:])**2
    
        # Print to screen
        string = 'EXT: real time propagation'
        pm.sprint(string, 1, newline=True)
        
        # Perform real time iterations
        for i in range(pm.sys.imax):

            # Construct the vector b
            b = C*wavefunction   

            # Solve Ax=b
            wavefunction, info = spsla.cg(A,b,x0=wavefunction,
                                 tol=pm.ext.rtol_solver)

            # Calculate the density
            density[i+1,:] = abs(wavefunction[:])**2

            # Calculate the norm of the wavefunction
            normalisation = (np.linalg.norm(wavefunction)*pm.sys.deltax**0.5)
            string = 'EXT: ' + 't = {:.5f}'.format((i+1)*pm.sys.deltat) + \
                     ', normalisation = ' + str(normalisation)
            pm.sprint(string, 1, newline=False)

        # Calculate the current density
        current_density = calculate_current_density(pm, density)

        # Save the time-dependent density, current density and external 
        # potential
        if(pm.run.time_dependence == True):
             results.add(density,'td_ext_den')
             results.add(current_density,'td_ext_cur')
             results.add(V,'td_ext_vxt')     
             if (pm.run.save):
                 results.save(pm)

    return results
