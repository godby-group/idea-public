"""Calculates the exact ground-state electron density and energy for a system
of three interacting electrons through solving the many-body Schrodinger 
equation. If the system is perturbed, the time-dependent electron density and 
current density are calculated. The ground-state and time-dependent ELF can 
also be calculated. Excited states of the unperturbed system can also be 
calculated.
"""
from __future__ import division
from __future__ import absolute_import

import time
import copy
import pickle
import numpy as np
import scipy as sp
import scipy.sparse as sps
import scipy.misc as spmisc
import scipy.special as spec
import scipy.linalg as spla
import scipy.sparse.linalg as spsla

from . import RE_cython
from . import EXT_cython
from . import NON	
from . import ELF 
from . import results as rs


def construct_antisymmetry_matrices(pm):
    r"""Constructs the reduction and expansion matrices that are used to
    exploit the exchange antisymmetry of the wavefunction.

    parameters
    ----------
    pm : object
        Parameters object

    returns sparse_matrix and sparse_matrix
        Reduction matrix used to reduce the wavefuction (remove indistinct
        elements). Expansion matrix used to expand the reduced wavefunction
        (insert indistinct elements) to get back the full wavefunction.
    """
    # Number of elements in the reduced wavefunction
    coo_size = int(np.prod(list(range(pm.sys.grid,pm.sys.grid+3)))\
               /spmisc.factorial(3))

    # COOrdinate holding arrays for the reduction matrix
    coo_1 = np.zeros((coo_size), dtype=int)
    coo_2 = np.copy(coo_1)
    coo_data_1 = np.ones((coo_size), dtype=np.float)

    # COOrdinate holding arrays for the expansion matrix
    coo_3 = np.zeros((pm.sys.grid**3), dtype=int)
    coo_4 = np.copy(coo_3)
    coo_data_2 = np.zeros((pm.sys.grid**3), dtype=np.float)

    # Populate the COOrdinate holding arrays with the coordinates and data
    coo_1, coo_2 = EXT_cython.reduction_three(coo_1, coo_2, pm.sys.grid)
    coo_3, coo_4, coo_data_2 = EXT_cython.expansion_three(coo_3, coo_4, 
                               coo_data_2, pm.sys.grid)

    # Convert the holding arrays into COOrdinate sparse matrices
    reduction_matrix = sps.coo_matrix((coo_data_1,(coo_1,coo_2)), shape=(
                       coo_size,pm.sys.grid**3), dtype=np.float)
    expansion_matrix = sps.coo_matrix((coo_data_2,(coo_3,coo_4)), shape=(
                       pm.sys.grid**3,coo_size), dtype=np.float)

    # Convert into compressed sparse row (csr) form for efficient arithemtic
    reduction_matrix = sps.csr_matrix(reduction_matrix)
    expansion_matrix = sps.csr_matrix(expansion_matrix)

    return reduction_matrix, expansion_matrix


def construct_A_reduced(pm, reduction_matrix, expansion_matrix, td):
    r"""Constructs the reduced form of the sparse matrix A.

    .. math:: 
        
        \text{Imaginary time}: \ &A = I + \dfrac{\delta \tau}{2}H \\
        \text{Real time}: \ &A = I + i\dfrac{\delta t}{2}H \\ \\
        &A_{\mathrm{red}} = RAE 

    where :math:`R =` reduction matrix and :math:`E =` expansion matrix

    parameters
    ----------
    pm : object
        Parameters object
    reduction_matrix : sparse_matrix
        Sparse_matrix used to reduce the wavefunction (remove indistinct 
        elements) by exploiting the exchange antisymmetry
    expansion_matrix : sparse_matrix
        Sparse_matrix used to expand the reduced wavefunction (insert 
        indistinct elements) to get back the full wavefunction
    td : integer
        0 for imaginary time, 1 for real time

    returns sparse_matrix
        Reduced form of the sparse matrix A, used when solving the equation 
        Ax=b
    """
    # Estimate the number of non-zero elements in the Hamiltonian matrix
    coo_size = int((3*pm.sys.stencil-2)*(pm.sys.grid**3))

    # COOrdinate holding arrays for the Hamiltonian matrix
    coo_1 = np.zeros((coo_size), dtype=int)
    coo_2 = np.copy(coo_1)
    coo_data = np.zeros((coo_size), dtype=np.float)

    # Pass the holding arrays and band elements to the Hamiltonian constructor, 
    # and populate the holding arrays with the coordinates and data
    coo_1, coo_2, coo_data = EXT_cython.hamiltonian_three(pm, coo_1, coo_2, coo_data, td)

    # Convert the holding arrays into a COOrdinate sparse matrix
    if(td == 0):
        prefactor = pm.ext.ideltat/2.0
        A = prefactor*sps.coo_matrix((coo_data,(coo_1,coo_2)), shape=(
            pm.sys.grid**3, pm.sys.grid**3), dtype=np.float)
        A += sps.identity(pm.sys.grid**3, dtype=np.float)
    elif(td == 1):
        coo_data = coo_data.astype(np.cfloat)
        prefactor = 1.0j*pm.sys.deltat/2.0
        A = prefactor*sps.coo_matrix((coo_data,(coo_1,coo_2)), shape=(pm.sys.grid**3,
            pm.sys.grid**3), dtype=np.cfloat)
        A += sps.identity(pm.sys.grid**3, dtype=np.cfloat)
        if(pm.sys.im == 1):
            imag_pot = EXT_cython.imag_pot_three(pm)
            A += prefactor*sps.spdiags(imag_pot, 0, pm.sys.grid**3, pm.sys.grid**3)

    # Convert into compressed sparse column (csc) form for efficient arithemtic
    A = sps.csc_matrix(A)

    # Construct the reduced form of A
    A_reduced = reduction_matrix*A*expansion_matrix
   
    return A_reduced


def initial_wavefunction(pm, ground_state=True):
    r"""Generates the initial wavefunction for the Crank-Nicholson imaginary 
    time propagation.

    .. math:: 

        \Psi(x_{1},x_{2},x_{3}) = \frac{1}{\sqrt{6}}\big(\phi_{1}(x_{1})
        \phi_{2}(x_{2})\phi_{3}(x_{3}) - \phi_{1}(x_{1})\phi_{3}(x_{2})
        \phi_{2}(x_{3}) + \phi_{3}(x_{1})\phi_{1}(x_{2})\phi_{2}(x_{3}) 
        - \phi_{3}(x_{1})\phi_{2}(x_{2})\phi_{1}(x_{3}) + \phi_{2}(x_{1})
        \phi_{3}(x_{2})\phi_{1}(x_{3}) - \phi_{2}(x_{1})\phi_{1}(x_{2})
        \phi_{3}(x_{3}) \big)

    parameters
    ----------
    pm : object
        Parameters object
    ground_state : bool
        - True: Construct a Slater determinant of either the three lowest 
                non-interacting eigenstates of the system, the three lowest
                Hartree-Fock eigenstates of the system or the three lowest
                LDA eigenstates of the system to use as the initial 
                wavefunction. Alternatively, read in a many-body wavefunction
                to use as the initial wavefunction.
        - False: Construct a Slater determinant of the three lowest eigenstates 
                 of the harmonic oscillator to use as the initial wavefunction

    returns array_like
        1D array of the reduced wavefunction, indexed as 
        wavefunction_reduced[space_index_1_2_3]
    """
    # Single-particle eigenstates
    eigenstate_1 = np.zeros(pm.sys.grid, dtype=np.float)
    eigenstate_2 = np.copy(eigenstate_1)
    eigenstate_3 = np.copy(eigenstate_1)

    # If calculating the ground-state wavefunction
    if(ground_state == True):

        # Read the three lowest Hartree-Fock eigenstates of the system
        if(pm.ext.initial_psi == 'hf'):
            try:
                eigenstates = rs.Results.read('gs_hf_eigf', pm) 
                eigenstate_1 = eigenstates[0].real
                eigenstate_2 = eigenstates[1].real
                eigenstate_3 = eigenstates[2].real

            # File does not exist
            except:
                raise IOError("Cannot find file containing HF orbitals.")

        # Read the three lowest LDA eigenstates of the system
        elif(pm.ext.initial_psi == 'lda'):
            try:
                eigenstates = rs.Results.read('gs_lda_eigf', pm)
                eigenstate_1 = eigenstates[0].real
                eigenstate_2 = eigenstates[1].real
                eigenstate_3 = eigenstates[2].real

            # File does not exist 
            except:
                raise IOError("Cannot find file containing LDA orbitals.")

        # Read the three lowest non-interacting eigenstates of the system
        elif(pm.ext.initial_psi == 'non'):
            try:
                eigenstates = rs.Results.read('gs_non_eigf', pm)
                eigenstate_1 = eigenstates[0].real
                eigenstate_2 = eigenstates[1].real
                eigenstate_3 = eigenstates[2].real

            # If the file does not exist, calculate the three lowest 
            # eigenstates
            except:
                eigenstate_1, eigenstate_2, eigenstate_3 = non_approx(pm)

        # Calculate the three lowest eigenstates of the harmonic oscillator
        elif(pm.ext.initial_psi == 'qho'):
            eigenstate_1 = qho_approx(pm, 0)
            eigenstate_2 = qho_approx(pm, 1)
            eigenstate_3 = qho_approx(pm, 2)

        # Read an exact many-body wavefunction from this directory 
        elif(pm.ext.initial_psi == 'ext'):
            try:
                wavefunction_reduced = rs.Results.read('gs_ext_psi', pm)
            
            # File does not exist
            except:
                raise IOError("Cannot find file containting many-body" + \
                " wavefunction.")

        # Read an exact many-body wavefunction from a different directory
        else:
            try:
                pm2 = copy.deepcopy(pm)
                pm2.run.name = pm.ext.initial_psi
                wavefunction_reduced = rs.Results.read('gs_ext_psi', pm2)
 
            # File does not exist
            except:
                raise IOError("Cannot find file containing many-body" + \
                " wavefunction.")

    # If calculating excited-state wavefunctions
    elif(ground_state == False): 

        # Calculate the three lowest eigenstates of the harmonic oscillator
        eigenstate_1 = qho_approx(pm, 0)
        eigenstate_2 = qho_approx(pm, 1)
        eigenstate_3 = qho_approx(pm, 2)

    # Construct a Slater determinant from the single-particle eigenstates if a
    # many-body wavefunction has not been read 
    nonzero_1 = np.count_nonzero(eigenstate_1)
    nonzero_2 = np.count_nonzero(eigenstate_2)
    nonzero_3 = np.count_nonzero(eigenstate_3)
    if(nonzero_1 != 0 and nonzero_2 != 0 and nonzero_3 != 0):
        wavefunction_reduced = EXT_cython.wavefunction_three(eigenstate_1, 
                               eigenstate_2, eigenstate_3)

    return wavefunction_reduced


def non_approx(pm):
    r"""Calculates the three lowest non-interacting eigenstates of the system.  
    These can then be expressed in Slater determinant form as an approximation  
    to the exact many-body wavefunction.

    parameters
    ----------
    pm : object
        Parameters object

    returns array_like and array_like and array_like
        1D array of the 1st non-interacting eigenstate, indexed as 
        eigenstate_1[space_index]. 1D array of the 2nd non-interacting 
        eigenstate, indexed as eigenstate_2[space_index]. 1D array of the 3rd
        non-interacting eigenstate, indexed as eigenstate_2[space_index].
    """
    # Construct the single-particle Hamiltonian
    K = NON.construct_K(pm)
    V = NON.construct_V(pm, 0)
    H = copy.copy(K)
    H[0,:] += V[:]

    # Solve the single-particle TISE
    eigenvalues, eigenfunctions = spla.eig_banded(H, lower=True, 
                                  select='i', select_range=(0,2))

    # Take the three lowest eigenstates
    eigenstate_1 = eigenfunctions[:,0]
    eigenstate_2 = eigenfunctions[:,1]
    eigenstate_3 = eigenfunctions[:,2]

    return eigenstate_1, eigenstate_2, eigenstate_3


def qho_approx(pm, n):
    r"""Calculates the nth energy eigenstate of the quantum harmonic 
    oscillator, and shifts to ensure it is neither an odd nor an even 
    function (necessary for the Gram-Schmidt algorithm). 

    parameters
    ----------
    pm : object
        Parameters object
    n : integer
        Principle quantum number

    returns array_like
        1D array of the nth eigenstate, indexed as eigenstate[space_index]
    """
    # Single-particle eigenstate
    eigenstate = np.zeros(pm.sys.grid, dtype=np.float)

    # Constants
    factorial = np.arange(0,n+1,1)
    fact = np.product(factorial[1:])
    norm = np.sqrt(1.0/((2.0**n)*fact)) / np.pi**0.25
    scale_factor = 7.0/pm.sys.xmax

    # Assign elements
    for j in range(pm.sys.grid):
        x = -pm.sys.xmax + j*pm.sys.deltax
        eigenstate[j] = (norm*(spec.hermite(n)(scale_factor*(x+1.0)))*(
                        0.25)*np.exp(-0.5*((scale_factor*(x+1.0))**2))) 

    return eigenstate


def calculate_energy(pm, wavefunction_reduced, wavefunction_reduced_old):
    r"""Calculates the energy of the system.

    .. math:: 

        E = - \ln\bigg(\dfrac{|\Psi(x_{1},x_{2},x_{3},\tau)|}{|\Psi(x_{1},
            x_{2},x_{3},\tau - \delta \tau)|}\bigg) \dfrac{1}{\delta \tau}

    parameters
    ----------
    pm : object
        Parameters object
    wavefunction_reduced : array_like
        1D array of the reduced wavefunction at t, indexed as 
        wavefunction_reduced[space_index_1_2_3]
    wavefunction_reduced_old : array_like
        1D array of the reduced wavefunction at t-dt, indexed as 
        wavefunction_reduced_old[space_index_1_2_3]

    returns float
        Energy of the system
    """
    a = np.linalg.norm(wavefunction_reduced_old)
    b = np.linalg.norm(wavefunction_reduced)
    energy = -np.log(b/a)/pm.ext.ideltat

    return energy


def calculate_density(pm, wavefunction_3D):
    r"""Calculates the electron density from the three-electron wavefunction.

    .. math:: 
   
        n(x) = 3 \int_{-x_{max}}^{x_{max}} \int_{-x_{max}}^{x_{max}} 
               |\Psi(x,x_{2},x_{3})|^2 dx_{2} dx_{3}

    parameters
    ----------
    pm : object
        Parameters object
    wavefunction : array_like
        3D array of the wavefunction, indexed as 
        wavefunction_3D[space_index_1,space_index_2,space_index_3]

    returns array_like
        1D array of the density, indexed as density[space_index]
    """
    mod_wavefunction_3D = np.absolute(wavefunction_3D)**2
    density = 3.0*np.sum(np.sum(mod_wavefunction_3D, axis=1, dtype=np.float), 
              axis=1)*pm.sys.deltax**2

    return density 


def calculate_current_density(pm, density):
    r"""Calculates the current density from the time-dependent 
    (and ground-state) electron density.

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
    string = 'EXT: calculating current density'
    pm.sprint(string, 1, newline=True)
    for i in range(1, pm.sys.imax):
         string = 'EXT: t = {:.5f}'.format(i*pm.sys.deltat)
         pm.sprint(string, 1, newline=False)
         J = np.zeros(pm.sys.grid, dtype=np.float)
         J = RE_cython.continuity_eqn(pm, density[i,:], density[i-1,:])
         current_density[i,:] = J[:]
    pm.sprint('', 1, newline=True)

    return current_density


def gram_schmidt(pm, wavefunction_reduced, eigenstates_array, excited_state):
    r"""Applies the Gram-Schmidt orthogonalisation process to project out 
    the eigenstates, that have already been calculated, from the 
    reduced wavefunction. This ensures that the imaginary time propagation
    converges to the next eigenstate.

    .. math:: 

        \Psi(x_{1},x_{2},x_{3},\tau) \rightarrow \Psi(x_{1},x_{2},x_{3},\tau) 
        - \sum_{j} (\Psi(x_{1},x_{2},x_{3},\tau)\boldsymbol{\cdot}\phi_{j}) 
        \phi_{j} \delta x^2
    
    parameters
    ----------
    pm : object
        Parameters object
    wavefunction_reduced : array_like
        1D array of the reduced wavefunction, indexed as 
        wavefunction_reduced[space_index_1_2_3]
    eigenstates_array : array_like
        2D array of the eigenstates of the system, indexed as 
        eigenstates_array[eigenstate,space_index_1_2_3] 
    excited_state : integer
        The excited state that is being calculated 

    returns array_like
        1D array of the reduced wavefunction after the eigenstates that have 
        already been calculated have been projected out. 
    """
    # Copy the reduced wavefunction
    v0 = wavefunction_reduced.copy()  
  
    # Project out the eigenstates
    for j in range(excited_state):
        vj = eigenstates_array[j,:]
        wavefunction_reduced -= (np.vdot(v0,vj))*vj*pm.sys.deltax**2
    
    return wavefunction_reduced


def solve_imaginary_time(pm, A_reduced, C_reduced, wavefunction_reduced, 
                         reduction_matrix, expansion_matrix, 
                         eigenstates_array=None):
    r"""Propagates the initial wavefunction through imaginary time using the 
    Crank-Nicholson method to find the ground-state of the system. 

    .. math:: 

        &\Big(I + \dfrac{\delta \tau}{2}H\Big) \Psi(x_{1},x_{2},x_{3},
        \tau+\delta \tau) = \Big(I - \dfrac{\delta \tau}{2}H\Big) 
        \Psi(x_{1},x_{2},x_{3},\tau) \\
        &\Psi(x_{1},x_{2},x_{3},\tau) = \sum_{m}c_{m}e^{-\varepsilon_{m}
        \tau}\phi_{m} \implies \lim_{\tau \to \infty} \Psi(x_{1},x_{2},
        x_{3},\tau) = \phi_{0}

    parameters
    ----------
    pm : object
        Parameters object
    A_reduced : sparse_matrix
        Reduced form of the sparse matrix A, used when solving the equation 
        Ax=b
    C_reduced : sparse_matrix
        Reduced form of the sparse matrix C, defined as C=-A+2I
    wavefunction_reduced : array_like
        1D array of the reduced wavefunction, indexed as 
        wavefunction_reduced[space_index_1_2_3]
    reduction_matrix : sparse_matrix
        Sparse matrix used to reduce the wavefunction (remove indistinct 
        elements) by exploiting the exchange antisymmetry
    expansion_matrix : sparse_matrix
        Sparse matrix used to expand the reduced wavefunction (insert 
        indistinct elements) to get back the full wavefunction
    eigenstates_array : array_like
        2D array of the eigenstates of the system, indexed as 
        eigenstates_array[eigenstate,space_index_1_2_3] 

    returns float and array_like
        Energy of the ground-state system. 1D array of the ground-state 
        wavefunction, indexed as wavefunction[space_index_1_2_3].
    """
    # Copy the initial wavefunction
    wavefunction_reduced_old = np.copy(wavefunction_reduced)

    # Print to screen
    string = 'EXT: imaginary time propagation'
    pm.sprint(string, 1, newline=True)

    # If calculating excited states
    if(eigenstates_array is not None):
        excited_state = eigenstates_array.shape[0]
  
    # Perform iterations
    i = 1
    while (i < pm.ext.iimax):

        # Begin timing the iteration
        start = time.time()
        string = 'imaginary time = {:.5f}'.format(i*pm.ext.ideltat) 
        pm.sprint(string, 0, newline=True)

        # Save the previous time step 
        wavefunction_reduced_old[:] = wavefunction_reduced[:]

        # Construct the reduction vector of b
        b_reduced = C_reduced*wavefunction_reduced

        # Solve Ax=b
        wavefunction_reduced, info = spsla.cg(A_reduced,b_reduced,
                                     x0=wavefunction_reduced,
                                     tol=pm.ext.itol_solver)

        # Apply Gram-Schmidt orthogonalisation if necessary
        if(eigenstates_array is not None):
            wavefunction_reduced = gram_schmidt(pm, wavefunction_reduced, 
                                   eigenstates_array, excited_state) 
 
        # Normalise the reduced wavefunction
        norm = np.linalg.norm(wavefunction_reduced)*(np.sqrt(pm.sys.deltax)**3)
        wavefunction_reduced[:] = wavefunction_reduced[:]/norm
        
        # Stop timing the iteration
        finish = time.time()
        string = 'time to complete step: {:.5f}'.format(finish - start)
        pm.sprint(string, 0, newline=True)

        # Calculate the convergence of the wavefunction
        wavefunction_convergence = np.linalg.norm(wavefunction_reduced_old - 
                                   wavefunction_reduced)
        string = 'wavefunction convergence: {}'.format(wavefunction_convergence)
        pm.sprint(string, 0, newline=True) 
        if(pm.run.verbosity=='default'):
            string = 'EXT: t = {:.5f}, convergence = {}' \
                    .format(i*pm.ext.ideltat, wavefunction_convergence)
            pm.sprint(string, 1, newline=False)
        if(wavefunction_convergence < pm.ext.itol):
            i = pm.ext.iimax
            pm.sprint('', 1, newline=True)
            if(eigenstates_array is None):
                string = 'EXT: ground-state converged' 
            else: 
                string = 'EXT: {} excited state converged'.format(
                         excited_state)
            pm.sprint(string, 1, newline=True)
        string = '--------------------------------------------------------' + \
                 '----------'
        pm.sprint(string, 0, newline=True)
        
        # Iterate
        i += 1

    # Calculate the energy
    wavefunction_reduced[:] = norm*wavefunction_reduced[:]
    energy = calculate_energy(pm, wavefunction_reduced, 
             wavefunction_reduced_old)
    if(eigenstates_array is None):
        string = 'EXT: ground-state energy = {:.5f}'.format(energy)
    else:
        string = 'EXT: {0} excited state energy = {1:.5f}'.format(
                 excited_state, energy)
    pm.sprint(string, 1, newline=True)
 
    # Expand the wavefunction and normalise
    wavefunction = expansion_matrix*wavefunction_reduced
    norm = np.linalg.norm(wavefunction)*(np.sqrt(pm.sys.deltax)**3)
    wavefunction[:] = wavefunction[:]/norm

    return energy, wavefunction


def solve_real_time(pm, A_reduced, C_reduced, wavefunction, reduction_matrix,
                    expansion_matrix):
    r"""Propagates the ground-state wavefunction through real time using the
    Crank-Nicholson method to find the time-evolution of the perturbed system.

    .. math:: 

        \Big(I + i\dfrac{\delta t}{2}H\Big) \Psi(x_{1},x_{2},x_{3},t+\delta t) = 
        \Big(I - i\dfrac{\delta t}{2}H\Big) \Psi(x_{1},x_{2},x_{3},t)   

    parameters
    ----------
    pm : object
        Parameters object
    A_reduced : sparse_matrix
        Reduced form of the sparse matrix A, used when solving the equation 
        Ax=b
    C_reduced : sparse_matrix
        Reduced form of the sparse matrix C, defined as C=-A+2I
    wavefunction : array_like
        1D array of the ground-state wavefunction, indexed as 
        wavefunction[space_index_1_2_3]
    reduction_matrix : sparse_matrix
        Sparse matrix used to reduce the wavefunction (remove indistinct
        elements) by exploiting the exchange antisymmetry
    expansion_matrix : sparse_matrix
        Sparse matrix used to expand the reduced wavefunction (insert
        indistinct elements) to get back the full wavefunction 

    returns array_like and array_like
        2D array of the time-dependent density, indexed as 
        density[time_index,space_index]. 2D array of the current density, 
        indexed as current_density[time_index,space_index]
    """
    # Array initialisations
    density = np.zeros((pm.sys.imax,pm.sys.grid), dtype=np.float)
    if(pm.ext.elf_td == 1):
        elf = np.copy(density)
    else:
        elf = 0 

    # Save the ground-state
    wavefunction_3D = wavefunction.reshape(pm.sys.grid, pm.sys.grid, 
                      pm.sys.grid)
    density[0,:] = calculate_density(pm, wavefunction_3D)
 
    # Reduce the wavefunction
    wavefunction_reduced = reduction_matrix*wavefunction

    # Print to screen
    string = 'EXT: real time propagation'
    pm.sprint(string, 1, newline=True)

    # Perform iterations
    for i in range(1, pm.sys.imax):
     
        # Begin timing the iteration
        start = time.time()
        string = 'real time = {:.5f}'.format(i*pm.sys.deltat) + '/' + \
                 '{:.5f}'.format((pm.sys.imax)*pm.sys.deltat)
        pm.sprint(string, 0, newline=True)
     
        # Construct the vector b and its reduction vector
        b_reduced = C_reduced*wavefunction_reduced
     
        # Solve Ax=b
        wavefunction_reduced, info = spsla.cg(A_reduced,b_reduced,
                                     x0=wavefunction_reduced,
                                     tol=pm.ext.rtol_solver)
      
        # Expand the wavefunction
        wavefunction = expansion_matrix*wavefunction_reduced
       
        # Calculate the density (and ELF)
        wavefunction_3D = wavefunction.reshape(pm.sys.grid, pm.sys.grid, 
                          pm.sys.grid)
        density[i,:] = calculate_density(pm, wavefunction_3D)
        if(pm.ext.elf_td == 1):
            elf[i,:] = ELF.main(pm, wavefunction_3D, density=density[i,:])
    
        # Stop timing the iteration
        finish = time.time()
        string = 'time to complete step: {:.5f}'.format(finish - start)
        pm.sprint(string, 0, newline=True)
      
        # Print to screen
        if(pm.run.verbosity == 'default'):
            string = 'EXT: ' + 't = {:.5f}'.format(i*pm.sys.deltat)
            pm.sprint(string, 1, newline=False)
        else:
            string = 'residual: {:.5f}'.format(np.linalg.norm(
                     A_reduced*wavefunction_reduced - b_reduced))
            pm.sprint(string, 0, newline=True)
            norm = np.linalg.norm(wavefunction)*(np.sqrt(pm.sys.deltax)**3)
            string = 'normalisation: {:.5f}'.format(norm**2)
            pm.sprint(string, 0, newline=True)
            string = '----------------------------------------------------' + \
                     '----------'
            pm.sprint(string, 0, newline=True)
    
    # Calculate the current density
    current_density = calculate_current_density(pm, density)

    if(pm.ext.elf_td == 1):
        return density, current_density, elf
    else:
        return density, current_density, elf


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
    # Array initialisations
    pm = parameters 
    string = 'EXT: constructing arrays'
    pm.sprint(string, 1, newline=True)
    pm.setup_space()

    # Construct the reduction and expansion matrices
    reduction_matrix, expansion_matrix = construct_antisymmetry_matrices(pm)

    # Construct the reduced form of the sparse matrices A and C
    A_reduced = construct_A_reduced(pm, reduction_matrix, expansion_matrix, 0)
    C_reduced = -A_reduced + 2.0*reduction_matrix*sps.identity(pm.sys.grid**3,
                dtype=np.float)*expansion_matrix

    # Generate the initial wavefunction
    wavefunction_reduced = initial_wavefunction(pm)

    # Propagate through imaginary time
    energy, wavefunction = solve_imaginary_time(pm, A_reduced, C_reduced,
                           wavefunction_reduced, reduction_matrix,
                           expansion_matrix) 
 
    # Calculate ground-state density
    wavefunction_3D = wavefunction.reshape(pm.sys.grid, pm.sys.grid, 
                      pm.sys.grid)
    density = calculate_density(pm, wavefunction_3D)
   
    # Save ground-state density, energy and external potential (optionally
    # the reduced wavefunction and ELF)
    results = rs.Results()
    results.add(density,'gs_ext_den')
    results.add(energy,'gs_ext_E')
    results.add(pm.space.v_ext,'gs_ext_vxt')
    if(pm.ext.psi_gs == 1):
        wavefunction_reduced = reduction_matrix*wavefunction
        results.add(wavefunction_reduced,'gs_ext_psi')
    if(pm.ext.elf_gs == 1):
        elf = ELF.main(pm, wavefunction_3D, density=density)
        results.add(elf,'gs_ext_elf')
    if(pm.run.save):
        results.save(pm)

    # Apply Gram-Schmidt orthogonalisation if necessary
    if(pm.ext.excited_states != 0):

        # Array to store the eigenstates
        eigenstates_array = np.zeros((pm.ext.excited_states+1, 
                            wavefunction_reduced.shape[0]), dtype=np.float)

        # Loop over each excited-state
        for j in range(pm.ext.excited_states):

            # Save the previously calculated eigenstate
            wavefunction_reduced = reduction_matrix*wavefunction
            eigenstates_array[j,:] = wavefunction_reduced[:]

            # Generate the initial wavefunction
            wavefunction_reduced = initial_wavefunction(pm, 
                                   wavefunction_reduced, 
                                   ground_state=False)
          
            # Propagate through imaginary time
            energy, wavefunction = solve_imaginary_time(pm, A_reduced,
                                   C_reduced, wavefunction_reduced, 
                                   reduction_matrix, expansion_matrix, 
                                   eigenstates_array=eigenstates_array[0:j+1,:])

            # Calculate the excited-state density
            wavefunction_3D = wavefunction.reshape(pm.sys.grid, pm.sys.grid, 
                              pm.sys.grid)
            density = calculate_density(pm, wavefunction_3D)

            # Save the excited-state density and energy (optionally the
            # reduced wavefunction and ELF)
            results.add(density,'es_ext_den{}'.format(j+1))
            results.add(energy,'es_ext_E{}'.format(j+1))
            if(pm.ext.psi_es == 1):
                wavefunction_reduced = reduction_matrix*wavefunction
                results.add(wavefunction_reduced,'es_ext_psi{}'.format(j+1))
            if(pm.ext.elf_es == 1):
                elf = ELF.main(pm, wavefunction_3D, density=density)
                results.add(elf,'es_ext_elf{}'.format(j+1))
            if(pm.run.save):
                results.save(pm)

    # Dispose of the reduced sparse matrices
    del A_reduced
    del C_reduced
        
    # Real time
    if(pm.run.time_dependence == True):

        # Array initialisations
        string = 'EXT: constructing arrays'
        pm.sprint(string, 1, newline=True)
        wavefunction = wavefunction.astype(np.cfloat)

        # Construct the reduced form of the sparse matrices A and C 
        A_reduced = construct_A_reduced(pm, reduction_matrix, expansion_matrix, 1)
        C_reduced = -A_reduced + 2.0*reduction_matrix*sps.identity(
                    pm.sys.grid**3, dtype=np.cfloat)*expansion_matrix

        # Propagate the ground-state wavefunction through real time
        density, current_density, elf = solve_real_time(pm, A_reduced, 
                                        C_reduced, wavefunction, 
                                        reduction_matrix, expansion_matrix)

        # Dispose of the reduced sparse matrices
        del A_reduced
        del C_reduced

        # Save time-dependent density, current density and external potential
        # (and ELF)
        results.add(density,'td_ext_den')
        results.add(current_density,'td_ext_cur')
        results.add(pm.space.v_ext+pm.space.v_pert,'td_ext_vxt')
        if(pm.ext.elf_td == 1):
            results.add(elf,'td_ext_elf')
        if(pm.run.save):
            results.save(pm)

    return results
