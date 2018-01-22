"""Calculates the exact ground-state electron density and energy for a system 
of one electron through solving the Schrodinger equation. If the system is 
perturbed, the time-dependent electron density and current density are 
calculated. 
"""
from __future__ import division
from __future__ import print_function

import os
import sys
import math
import copy
import pickle
import numpy as np
import scipy as sp
import scipy.sparse as sps
import scipy.sparse.linalg as spsla
from . import RE_cython
from . import results as rs


def construct_K(pm):
    r"""Constructs the kinetic energy operator K on the system's grid. This is 
    constructed using a three-point, five-point or seven-point stencil, 
    yielding an NxN band matrix (where N is the number of grid points). For 
    example with N=6 and a three-point stencil:
   
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
        K = -0.5*sps.diags([1, -2, 1],[-1, 0, 1], shape=(pm.sys.grid,
        pm.sys.grid), format='csr')/(pm.sys.deltax**2)
    elif(pm.sys.stencil == 5):
        K = -0.5*sps.diags([-1.0/12.0, 4.0/3.0, -2.5, 4.0/3.0, -1.0/12.0],
        [-2, -1, 0, 1, 2], shape=(pm.sys.grid,pm.sys.grid), format='csr')/(
        pm.sys.deltax**2)
    elif(pm.sys.stencil == 7):
        K = -0.5*sps.diags([1.0/90.0, -3.0/20.0, 3.0/2.0, -49.0/18.0, 3.0/2.0, 
        -3.0/20.0, 1.0/90.0],[-3, -2, -1, 0, 1, 2, 3], shape=(pm.sys.grid,
        pm.sys.grid), format='csr')/(pm.sys.deltax**2)
    else:
        raise ValueError("pm.sys.stencil must be either 3, 5 or 7.")    

    return K


def construct_V(pm, td):
    r"""Constructs the potential energy operator V on the system's grid.
   
    V will contain V(x), sampled on the system's grid along the diagonal, 
    yielding an NxN diagonal matrix (where N is the number of grid points).

    parameters
    ----------
    pm : object
        Parameters object
    td : bool
         - 'False': Construct external potential
         - 'True': Construct peturbed potential

    returns sparse_matrix
        Potential energy matrix
    """
    xgrid = np.linspace(-pm.sys.xmax,pm.sys.xmax,pm.sys.grid)
    V_diagonal = np.empty(pm.sys.grid)

    if(td == 0):
        V_diagonal[:] = pm.sys.v_ext(xgrid[:])
    if(td == 1):
        V_diagonal[:] = (pm.sys.v_ext(xgrid[:]) + pm.sys.v_pert(xgrid[:]))
    V = sps.spdiags(V_diagonal, 0, pm.sys.grid, pm.sys.grid, format='csr')

    return V


def construct_A(pm, H, td):
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
    td : bool
         - 'False': Construct for imaginary-time propagation
         - 'True': Construct for real-time propagation
 
    returns sparse_matrix
        Sparse matrix A
    """
    I = sps.identity(pm.sys.grid)

    if(td == 0):   
        A = I + 1.0*(pm.ext.ideltat/2.0)*H
    if(td == 1):   
        A = I + 1.0j*(pm.sys.deltat/2.0)*H

    return A


def construct_C(pm, H, td):
    r"""Constructs the matrix C to be used when solving Ax=b (b=Cd) in the 
    Crank-Nicholson propagation.
    .. math::

        C = I - i \frac{dt}{2} H

    parameters
    ----------
    pm : object
        Parameters object
    H : sparse_matrix
        The Hamiltonian matrix
    td : bool
         - 'False': Construct for imaginary-time propagation
         - 'True': Construct for real-time propagation

    returns sparse_matrix
        Sparse matrix C
    """
    I = sps.identity(pm.sys.grid)

    if(td == 0):   
        C = I - 1.0*(pm.ext.ideltat/2.0)*H
    if(td == 1):   
        C = I - 1.0j*(pm.sys.deltat/2.0)*H

    return C


def calculate_energy(pm, H, wavefunction):
    r"""Calculates the energy of a given single-particle wavefunction.

    .. math::

        E = \langle \psi | H | \psi \rangle

    parameters
    ----------
    pm : object
        Parameters object
    H : sparse_matrix
        The Hamiltonian matrix
    wavefunction : array_like
        Single-particle wavefunction 

    returns double
        Energy
    """
    A = H*wavefunction
    B = wavefunction
    energy = 0.0
    for i in range(0,len(A)-1):
        energy += (B[i+1]*A[i+1]+B[i]*A[i])*pm.sys.deltax/2.0

    return energy.real
 

def calculate_density(pm, wavefunction):
    r"""Calculates the electron density from a given wavefunction.

    .. math::

        n \left(x \right) = |\psi \left( x\right)|^2

    parameters
    ----------
    pm : object
        Parameters object
    wavefunction : array_like
        The wavefunction

    returns array_like
        The electron density
    """
    density = np.empty(pm.sys.grid)
    density[:] = abs(wavefunction[:])**2

    return density


def calculate_current_density(pm, densities):
    r"""Calculates the current density of a time evolving wavefunction by 
    solving the continuity equation.

    .. math::

        \frac{\partial n}{\partial t} + \nabla \cdot j = 0

    parameters
    ----------
    pm : object
        Parameters object
    densities : array_like
        1D array of the time-dependent density, indexed as       
        density[time_index,space_index]

    returns array_like
        2D array of the current density, indexed as 
        current_density[time_index,space_index]
    """
    pm.sprint('', 1, newline=True)
    current_density = np.zeros((pm.sys.imax,pm.sys.grid), dtype=np.float)
    string = 'EXT: calculating current density'
    pm.sprint(string, 1, newline=True)
    for i in range(1, pm.sys.imax):
         string = 'EXT: t = {:.5f}'.format(i*pm.sys.deltat)
         pm.sprint(string, 1, newline=False)
         J = np.zeros(pm.sys.grid, dtype=np.float)
         J = RE_cython.continuity_eqn(pm, densities[i,:], densities[i-1,:]) 
         current_density[i,:] = J[:]
    pm.sprint('', 1, newline=True)

    return current_density


def initial_wavefunction(pm, x):
    r"""Calculates the value of the initial wavefunction at a given x.


    parameters
    ----------
    pm : object
        Parameters object
    x : float
       given x

    returns array_like
       Initial guess for wavefunction
    """
    return 1.0 / math.sqrt(2.0*math.pi) * math.e**(-0.5*x**2)


def main(parameters):
    r"""Performs calculation of the one-electron density

    parameters
    ----------
    parameters : object
        Parameters object

    returns object
        Results object
    """
    pm = parameters

    # Construct the grid
    x_points = np.linspace(-pm.sys.xmax,pm.sys.xmax,pm.sys.grid)

    # Construct the kinetic energy matrix
    K = construct_K(pm)
   
    # Construct the potential energy matrix
    V = construct_V(pm, 0)

    # Construct the Hamiltonian matrix, and the matrices used in the 
    # Crank-Nicholson propagation.
    H = K + V
    A = construct_A(pm, H, False)
    C = construct_C(pm, H, False)

    # Construct the initial wavefunction
    wavefunction = initial_wavefunction(pm, x_points)
   
    # Perform complex time iterations until converged
    i = 1
    while(i < pm.ext.iimax):

        # Construct the vector b
        b = C*wavefunction   
      
        # Set the previous time step
        wavefunction_old = wavefunction

        # Solve Ax=b
        wavefunction, info = spsla.cg(A,b,x0=wavefunction,tol=pm.ext.itol_solver)

        # Normalise the wavefunction
        wavefunction = wavefunction/(np.linalg.norm(wavefunction)*pm.sys.deltax**0.5)

        # Calculate the convergence of the wavefunction
        wavefunction_convergence = np.linalg.norm(wavefunction_old-wavefunction)
        string = 'EXT: imaginary time, t = {}, convergence = {}'\
                    .format(i*pm.ext.ideltat, wavefunction_convergence)
        pm.sprint(string, 1, newline=False)
        if(wavefunction_convergence < pm.ext.itol):
            i = pm.ext.iimax
            string = 'EXT: ground-state converged' 
            pm.sprint(string, 1, newline=True)
      
        # Iterate
        i += 1

    # Calculate the density
    density = calculate_density(pm, wavefunction)

    # Calculate the ground-state energy
    energy = calculate_energy(pm, H, wavefunction)
    print('EXT: ground-state energy =', energy)

    # Save the ground-state density, energy and external potential
    results = rs.Results()
    results.add(density, 'gs_ext_den')
    results.add(energy, 'gs_ext_E')
    results.add(V, 'gs_ext_vxt')
    if (pm.run.save):
        results.save(pm)

    # Propagate through real time
    if(pm.run.time_dependence == True):

        # Construct the potential energy matrix
        V = construct_V(pm, 1)

        # Construct the Hamiltonian matrix, and the matrices used in the 
        # Crank-Nicholson propagation.
        H = K + V
        A = construct_A(pm, H, True)
        C = construct_C(pm, H, True)

        # Construct time-dependent density array and save the ground-state
        densities = np.zeros((pm.sys.imax,pm.sys.grid), dtype=np.float)
        densities[0,:] = density

        # Convert wavefunction array to complex
        wavefunction = wavefunction.astype(np.cfloat)

        # Perform real time iterations
        for i in range(1, pm.sys.imax):

            # Construct the vector b
            b = C*wavefunction   

            # Solve Ax=b
            wavefunction, info = spsla.cg(A,b,x0=wavefunction,
                                 tol=pm.ext.rtol_solver)

            # Calculate the density
            densities[i,:] = calculate_density(pm, wavefunction)

            # Calculate the norm of the wavefunction
            normalisation = (np.linalg.norm(wavefunction)*pm.sys.deltax**0.5)
            string = 'EXT: real time, t = {}, normalisation = {}'\
                      .format(i*pm.sys.deltat, normalisation)
            pm.sprint(string, 1, newline=False)

        # Calculate the current density
        current_density = calculate_current_density(pm, densities)

        # Save the time-dependent density, current density and external potential
        if(pm.run.time_dependence == True):
             results.add(densities,'td_ext_den')
             results.add(current_density,'td_ext_cur')
             results.add(V,'td_ext_vxt')     
             if (pm.run.save):
                 results.save(pm)
      
    # Program complete
    os.system('rm *.pyc')

    return results

