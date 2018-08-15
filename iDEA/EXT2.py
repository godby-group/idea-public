"""Calculates the exact ground-state electron density and energy for a system  of two interacting
electrons through solving the many-electron Schroedinger equation. If the system is perturbed, the 
time-dependent electron density and current density are calculated. 
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import time
import copy
import numpy as np
import numpy.linalg as npla
import scipy as sp
import scipy.sparse as sps
import scipy.special as spspec
import scipy.linalg as spla
import scipy.sparse.linalg as spsla

from . import EXT_cython
from . import NON 
from . import results as rs


def construct_antisymmetry_matrices(pm):
    r"""Constructs the reduction and expansion matrices that are used to exploit the exchange 
    antisymmetry of the wavefunction.

    .. math::

        \Psi(x_{1},x_{2}) = -\Psi(x_{2},x_{1})

    parameters
    ----------
    pm : object
        Parameters object

    returns sparse_matrix and sparse_matrix
        Reduction matrix used to reduce the wavefunction (remove indistinct elements). 
        Expansion matrix used to expand the reduced wavefunction (insert indistinct elements) to 
        get back the full wavefunction.
    """
    # Number of elements in the reduced wavefunction
    coo_size = int(round(np.prod(list(range(pm.space.npt,pm.space.npt+2)))/spspec.factorial(2)))

    # COOrdinate holding arrays for the reduction matrix
    coo_1 = np.zeros(coo_size, dtype=int)
    coo_2 = np.copy(coo_1)
    coo_data_1 = np.ones(coo_size, dtype=np.float)
    
    # COOrdinate holding arrays for the expansion matrix 
    coo_3 = np.zeros(pm.space.npt**2, dtype=int)
    coo_4 = np.copy(coo_3)
    coo_data_2 = np.zeros(pm.space.npt**2, dtype=np.float)  

    # Populate the COOrdinate holding arrays with the coordinates and data
    coo_1, coo_2 = EXT_cython.reduction_two(pm, coo_1, coo_2)
    coo_3, coo_4, coo_data_2 = EXT_cython.expansion_two(pm, coo_3, coo_4, coo_data_2)

    # Convert the holding arrays into COOrdinate sparse matrices
    reduction_matrix = sps.coo_matrix((coo_data_1,(coo_1,coo_2)), shape=(coo_size,pm.space.npt**2), dtype=np.float)
    expansion_matrix = sps.coo_matrix((coo_data_2,(coo_3,coo_4)), shape=(pm.space.npt**2,coo_size), dtype=np.float)

    # Convert into compressed sparse row (csr) form for efficient arithemtic
    reduction_matrix = sps.csr_matrix(reduction_matrix)
    expansion_matrix = sps.csr_matrix(expansion_matrix)

    return reduction_matrix, expansion_matrix


def construct_A_reduced(pm, reduction_matrix, expansion_matrix, td):
    r"""Constructs the reduced form of the sparse matrix A.

    .. math:: 
        
            \text{Imaginary time}: \ &A = I + \frac{\delta \tau}{2}H \\
            \text{Real time}: \ &A = I + i\frac{\delta t}{2}H \\ \\
            &A_{\mathrm{red}} = RAE 

    where :math:`R =` reduction matrix and :math:`E =` expansion matrix

    parameters
    ----------
    pm : object
        Parameters object
    reduction_matrix : sparse_matrix
        Sparse matrix used to reduce the wavefunction (remove indistinct elements) by exploiting 
        the exchange antisymmetry
    expansion_matrix : sparse_matrix
        Sparse matrix used to expand the reduced wavefunction (insert indistinct elements) to get
        back the full wavefunction
    td : integer
        0 for imaginary time, 1 for real time

    returns sparse_matrix
        Reduced form of the sparse matrix A, used when solving the equation Ax=b
    """
    # Estimate the number of non-zero elements in the Hamiltonian matrix
    coo_size = int(round((2*pm.sys.stencil-1)*(pm.space.npt**2)))

    # COOrdinate holding arrays for the Hamiltonian matrix
    coo_1 = np.zeros(coo_size, dtype=int)
    coo_2 = np.copy(coo_1)
    coo_data = np.zeros(coo_size, dtype=np.float)

    # Pass the holding arrays and band elements to the Hamiltonian constructor, and populate the 
    # holding arrays with the coordinates and data
    coo_1, coo_2, coo_data = EXT_cython.hamiltonian_two(pm, coo_1, coo_2, coo_data, td)

    # Convert the holding arrays into a COOrdinate sparse matrix
    if(td == 0):
        prefactor = pm.ext.ideltat/2.0
        A = prefactor*sps.coo_matrix((coo_data,(coo_1,coo_2)), shape=(pm.space.npt**2,pm.space.npt**2), dtype=np.float)
        A += sps.identity(pm.space.npt**2, dtype=np.float)
    elif(td == 1):
        coo_data = coo_data.astype(np.cfloat)
        prefactor = 1.0j*pm.sys.deltat/2.0
        A = prefactor*sps.coo_matrix((coo_data,(coo_1,coo_2)), shape=(pm.space.npt**2,pm.space.npt**2), dtype=np.cfloat)
        A += sps.identity(pm.space.npt**2, dtype=np.cfloat)
        if(pm.sys.im == 1):
            imag_pot = EXT_cython.imag_pot_two(pm)
            A += prefactor*sps.spdiags(imag_pot, 0, pm.space.npt**2, pm.space.npt**2)
            
    # Convert into compressed sparse column (csc) form for efficient arithemtic
    A = sps.csc_matrix(A)
    
    # Construct the reduced form of A
    A_reduced = reduction_matrix*A*expansion_matrix

    return A_reduced


def initial_wavefunction(pm):
    r"""Generates the initial condition for the Crank-Nicholson imaginary time propagation.

    .. math:: 

        \Psi(x_{1},x_{2}) = \frac{1}{\sqrt{2}}\big(\phi_{1}(x_{1})\phi_{2}(x_{2}) 
        - \phi_{2}(x_{1})\phi_{1}(x_{2})\big)

    parameters
    ----------
    pm : object
        Parameters object

    returns array_like
        1D array of the reduced wavefunction, indexed as wavefunction_reduced[space_index_1_2]
    """
    # Single-electron eigenstates
    eigenstate_1 = np.zeros(pm.space.npt, dtype=np.float)
    eigenstate_2 = np.copy(eigenstate_1)

    # Read the two lowest Hartree-Fock eigenstates of the system
    if(pm.ext.initial_gspsi == 'hf'):
        try:
            eigenstates = rs.Results.read('gs_hf_eigf', pm) 
            eigenstate_1 = eigenstates[0].real
            eigenstate_2 = eigenstates[1].real
        # File does not exist
        except:
            raise IOError("EXT: cannot find file containing HF orbitals.")

    # Read the two lowest one-electron LDA eigenstates of the system
    elif(pm.ext.initial_gspsi == 'lda1'):
        try:
            eigenstates = rs.Results.read('gs_lda1_eigf', pm)
            eigenstate_1 = eigenstates[0].real
            eigenstate_2 = eigenstates[1].real
        # File does not exist 
        except:
            raise IOError("EXT: cannot find file containing one-electron LDA orbitals.")

    # Read the two lowest two-electron LDA eigenstates of the system
    elif(pm.ext.initial_gspsi == 'lda2'):
        try:
            eigenstates = rs.Results.read('gs_lda2_eigf', pm)
            eigenstate_1 = eigenstates[0].real
            eigenstate_2 = eigenstates[1].real
        # File does not exist 
        except:
            raise IOError("EXT: cannot find file containing two-electron LDA orbitals.")

    # Read the two lowest three-electron LDA eigenstates of the system
    elif(pm.ext.initial_gspsi == 'lda3'):
        try:
            eigenstates = rs.Results.read('gs_lda3_eigf', pm)
            eigenstate_1 = eigenstates[0].real
            eigenstate_2 = eigenstates[1].real
        # File does not exist 
        except:
            raise IOError("EXT: cannot find file containing three-electron LDA orbitals.")

    # Read the two lowest HEG LDA eigenstates of the system
    elif(pm.ext.initial_gspsi == 'ldaheg'):
        try:
            eigenstates = rs.Results.read('gs_ldaheg_eigf', pm)
            eigenstate_1 = eigenstates[0].real
            eigenstate_2 = eigenstates[1].real
        # File does not exist 
        except:
            raise IOError("EXT: cannot find file containing HEG LDA orbitals.")

    # Read the two lowest non-interacting eigenstates of the system
    elif(pm.ext.initial_gspsi == 'non'):
        try:
            eigenstates = rs.Results.read('gs_non_eigf', pm)
            eigenstate_1 = eigenstates[0].real
            eigenstate_2 = eigenstates[1].real
        # If the file does not exist, calculate the two lowest eigenstates
        except:
            eigenstate_1, eigenstate_2 = non_approx(pm)

    # Calculate the two lowest eigenstates of the harmonic oscillator
    elif(pm.ext.initial_gspsi == 'qho'):
        eigenstate_1 = qho_approx(pm, 0)
        eigenstate_2 = qho_approx(pm, 1)

    # Read an exact many-electron wavefunction from this directory 
    elif(pm.ext.initial_gspsi == 'ext'):
        try:
            wavefunction_reduced = rs.Results.read('gs_ext_psi', pm)          
        # File does not exist
        except:
            raise IOError("EXT: cannot find file containting many-electron wavefunction.")

    # Read an exact many-electron wavefunction from a different directory
    else:
        try:
            pm2 = copy.deepcopy(pm)
            pm2.run.name = pm.ext.initial_gspsi
            wavefunction_reduced = rs.Results.read('gs_ext_psi', pm2)
        # File does not exist
        except:
            raise IOError("EXT: cannot find file containing many-electron wavefunction.")

    # Construct a Slater determinant from the single-electron eigenstates if a many-electron wavefunction has not been read 
    nonzero_1 = np.count_nonzero(eigenstate_1)
    nonzero_2 = np.count_nonzero(eigenstate_2)
    if(nonzero_1 != 0 and nonzero_2 != 0):
        wavefunction_reduced = EXT_cython.wavefunction_two(pm, eigenstate_1, eigenstate_2)

    return wavefunction_reduced


def non_approx(pm):
    r"""Calculates the two lowest non-interacting eigenstates of the system. These can then be 
    expressed in Slater determinant form as an approximation to the exact many-electron wavefunction.

    .. math:: 

        \bigg(-\frac{1}{2} \frac{d^{2}}{dx^{2}} + V_{\mathrm{ext}}(x) \bigg) \phi_{j}(x) 
        = \varepsilon_{j} \phi_{j}(x)

    parameters
    ----------
    pm : object
        Parameters object

    returns array_like and array_like
        1D array of the 1st non-interacting eigenstate, indexed as eigenstate_1[space_index]. 
        1D array of the 2nd non-interacting eigenstate, indexed as eigenstate_2[space_index].
    """
    # Construct the single-electron Hamiltonian
    K = NON.construct_K(pm)
    H = copy.copy(K)
    H[0,:] += pm.space.v_ext[:]

    # Solve the single-electron TISE
    eigenvalues, eigenfunctions = spla.eig_banded(H, lower=True, select='i', select_range=(0,1))

    # Take the two lowest eigenstates
    eigenstate_1 = eigenfunctions[:,0]
    eigenstate_2 = eigenfunctions[:,1]

    return eigenstate_1, eigenstate_2


def qho_approx(pm, n):
    r"""Calculates the nth eigenstate of the quantum harmonic oscillator, and shifts to ensure it
    is neither an odd nor an even function (necessary for the Gram-Schmidt algorithm). 

    .. math:: 

        \bigg(-\frac{1}{2} \frac{d^{2}}{dx^{2}} + \frac{1}{2} \omega^{2} x^{2} \bigg) \phi_{n}(x) 
        = \varepsilon_{n} \phi_{n}(x)

        \phi_{n}(x) = \frac{1}{\sqrt{2^{n}n!}} \bigg(\frac{\omega}{\pi}\bigg)^{1/4} e^{-\frac{\omega x^{2}}{2}} 
        H_{n}\bigg(\sqrt{\omega}x \bigg)

    parameters
    ----------
    pm : object
        Parameters object
    n : integer
        Principle quantum number

    returns array_like
        1D array of the nth eigenstate, indexed as eigenstate[space_index]
    """
    # Single-electron eigenstate
    eigenstate = np.zeros(pm.space.npt, dtype=np.float)

    # Constants
    factorial = spspec.factorial(n)
    omega = 30.0/(pm.sys.xmax**2)
    norm = np.sqrt(np.sqrt(omega/np.pi)/((2.0**n)*factorial))

    # Assign elements
    for j in range(pm.space.npt):
        x = -pm.sys.xmax + j*pm.space.delta
        eigenstate[j] = norm*(spspec.hermite(n)(np.sqrt(omega)*(x+1.0)))*np.exp(-0.5*omega*((x+1.0)**2)) 

    return eigenstate


def calculate_energy(pm, wavefunction_reduced, wavefunction_reduced_old):
    r"""Calculates the energy of the system.

    .. math:: 

        E = - \ln\bigg(\frac{|\Psi(x_{1},x_{2},\tau)|}{|\Psi(x_{1},x_{2},\tau 
            - \delta \tau)|}\bigg) \frac{1}{\delta \tau}
 
    parameters
    ----------
    pm : object
        Parameters object
    wavefunction_reduced : array_like
        1D array of the reduced wavefunction at t, indexed as wavefunction_reduced[space_index_1_2]
    wavefunction_reduced_old : array_like
        1D array of the reduced wavefunction at t-dt, indexed as wavefunction_reduced_old[space_index_1_2]

    returns float
        Energy of the system
    """
    a = npla.norm(wavefunction_reduced_old)
    b = npla.norm(wavefunction_reduced)
    energy = -np.log(b/a)/pm.ext.ideltat

    return energy


def calculate_density(pm, wavefunction_2D):
    r"""Calculates the electron density from the two-electron wavefunction.

    .. math:: 
   
        n(x) = 2 \int_{-x_{\mathrm{max}}}^{x_{\mathrm{max}}} |\Psi(x,x_{2})|^{2} dx_{2}

    parameters
    ----------
    pm : object
        Parameters object
    wavefunction : array_like
        2D array of the wavefunction, indexed as wavefunction_2D[space_index_1,space_index_2]

    returns array_like
        1D array of the density, indexed as density[space_index]
    """
    mod_wavefunction_2D = np.absolute(wavefunction_2D)**2
    density = 2.0*np.sum(mod_wavefunction_2D, axis=1, dtype=np.float)*pm.space.delta

    return density 


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


def solve_imaginary_time(pm, A_reduced, C_reduced, wavefunction_reduced, expansion_matrix):
    r"""Propagates the initial wavefunction through imaginary time using the Crank-Nicholson method 
    to find the ground-state of the system.

    .. math:: 

        &\Big(I + \frac{\delta \tau}{2}H\Big) \Psi(x_{1},x_{2},\tau+\delta \tau)
        = \Big(I - \frac{\delta \tau}{2}H\Big) \Psi(x_{1},x_{2},\tau) \\
        &\Psi(x_{1},x_{2},\tau) = \sum_{m}c_{m}e^{-\varepsilon_{m} \tau}\phi_{m} 
        \implies \lim_{\tau \to \infty} \Psi(x_{1},x_{2},\tau) = \phi_{0}

    parameters
    ----------
    pm : object
        Parameters object
    A_reduced : sparse_matrix
        Reduced form of the sparse matrix A, used when solving the equation Ax=b
    C_reduced : sparse_matrix
        Reduced form of the sparse matrix C, defined as C=-A+2I
    wavefunction_reduced : array_like
        1D array of the reduced wavefunction, indexed as wavefunction_reduced[space_index_1_2]
    expansion_matrix : sparse_matrix
        Sparse matrix used to expand the reduced wavefunction (insert indistinct elements) to get 
        back the full wavefunction

    returns float and array_like
        Energy of the ground-state system. 1D array of the ground-state wavefunction, indexed as 
        wavefunction[space_index_1_2].
    """
    # Copy the initial wavefunction
    wavefunction_reduced_old = np.copy(wavefunction_reduced)
  
    # Print to screen
    string = 'EXT: imaginary time propagation'
    pm.sprint(string, 1)

    # Perform iterations
    i = 1
    while (i < pm.ext.iimax):

        # Begin timing the iteration
        start = time.time()
        string = 'imaginary time = {:.5f}'.format(i*pm.ext.ideltat) 
        if(i % 1000 == 0):
            pm.sprint(string, 0)
        else:
            pm.sprint(string, 0, savelog=False)

        # Save the previous time step 
        wavefunction_reduced_old[:] = wavefunction_reduced[:]
        
        # Construct the reduction vector of b
        b_reduced = C_reduced*wavefunction_reduced
      
        # Solve Ax=b
        wavefunction_reduced, info = spsla.cg(A_reduced, b_reduced, x0=wavefunction_reduced, tol=pm.ext.itol_solver) 

        # Normalise the reduced wavefunction
        norm = npla.norm(wavefunction_reduced)*pm.space.delta
        wavefunction_reduced /= norm
           
        # Stop timing the iteration
        finish = time.time()
        string = 'time to complete step: {:.5f}'.format(finish - start)
        if(i % 1000 == 0):
            pm.sprint(string, 0)
        else:
            pm.sprint(string, 0, savelog=False)

        # Calculate the convergence of the wavefunction
        wavefunction_convergence = npla.norm(wavefunction_reduced_old - wavefunction_reduced)
        string = 'wavefunction convergence: {}'.format(wavefunction_convergence)
        if(i % 1000 == 0):
            pm.sprint(string, 0)
        else:
            pm.sprint(string, 0, savelog=False) 
        if(pm.run.verbosity == 'default'):
            string = 'EXT: t = {:.5f}, convergence = {}'.format(i*pm.ext.ideltat, wavefunction_convergence)
            if(i % 1000 == 0):
                pm.sprint(string, 1, newline=False)
            else:
                pm.sprint(string, 1, newline=False, savelog=False)
        if(wavefunction_convergence < pm.ext.itol):
            i = pm.ext.iimax
            pm.sprint('', 1)
            string = 'EXT: ground-state converged' 
            pm.sprint(string, 1)
        string = '------------------------------------------------------------------'
        if(i % 1000 == 0):
            pm.sprint(string, 0)
        else:
            pm.sprint(string, 0, savelog=False)

        # Iterate
        i += 1

    # Calculate the energy
    wavefunction_reduced *= norm
    energy = calculate_energy(pm, wavefunction_reduced, wavefunction_reduced_old)
    string = 'EXT: ground-state energy = {:.5f}'.format(energy)
    pm.sprint(string, 1)

    # Expand the wavefunction and normalise
    wavefunction = expansion_matrix*wavefunction_reduced
    norm = npla.norm(wavefunction)*pm.space.delta
    wavefunction /= norm

    return energy, wavefunction


def solve_real_time(pm, A_reduced, C_reduced, wavefunction, reduction_matrix, expansion_matrix):
    r"""Propagates the ground-state wavefunction through real time using the Crank-Nicholson method 
    to find the time-evolution of the perturbed system.

    .. math:: 

        \Big(I + i\frac{\delta t}{2}H\Big) \Psi(x_{1},x_{2},t+\delta t) = 
        \Big(I - i\frac{\delta t}{2}H\Big) \Psi(x_{1},x_{2},t)   

    parameters
    ----------
    pm : object
        Parameters object
    A_reduced : sparse_matrix
        Reduced form of the sparse matrix A, used when solving the equation Ax=b
    C_reduced : sparse_matrix
        Reduced form of the sparse matrix C, defined as C=-A+2I
    wavefunction : array_like
        1D array of the ground-state wavefunction, indexed as wavefunction[space_index_1_2]
    reduction_matrix : sparse_matrix
        Sparse matrix used to reduce the wavefunction (remove indistinct elements) by exploiting 
        the exchange antisymmetry
    expansion_matrix : sparse_matrix
        Sparse matrix used to expand the reduced wavefunction (insert indistinct elements) to get 
        back the full wavefunction

    returns array_like and array_like
        2D array of the time-dependent density, indexed as density[time_index,space_index]. 
        2D array of the current density, indexed as current_density[time_index,space_index].
    """
    # Array initialisations
    density = np.zeros((pm.sys.imax,pm.space.npt), dtype=np.float)

    # Save the ground-state
    wavefunction_2D = wavefunction.reshape(pm.space.npt, pm.space.npt)
    density[0,:] = calculate_density(pm, wavefunction_2D)
 
    # Reduce the wavefunction
    wavefunction_reduced = reduction_matrix*wavefunction

    # Print to screen
    string = 'EXT: real time propagation'
    pm.sprint(string, 1)

    # Perform iterations
    for i in range(1, pm.sys.imax):

        # Begin timing the iteration
        start = time.time()
        string = 'real time = {:.5f}'.format(i*pm.sys.deltat) + '/' + '{:.5f}'.format((pm.sys.imax)*pm.sys.deltat)
        if(i % 100 == 0):
            pm.sprint(string, 0)
        else:
            pm.sprint(string, 0, savelog=False)

        # Construct the vector b and its reduction vector
        b_reduced = C_reduced*wavefunction_reduced

        # Solve Ax=b
        wavefunction_reduced, info = spsla.cg(A_reduced, b_reduced, x0=wavefunction_reduced, tol=pm.ext.rtol_solver)

        # Expand the wavefunction and normalise
        wavefunction = expansion_matrix*wavefunction_reduced
        norm = npla.norm(wavefunction)*pm.space.delta
        if(pm.sys.im == 0):
            wavefunction /= norm
            
        # Calculate the density 
        wavefunction_2D = wavefunction.reshape(pm.space.npt, pm.space.npt)
        density[i,:] = calculate_density(pm, wavefunction_2D)
  
        # Stop timing the iteration
        finish = time.time()
        string = 'time to complete step: {:.5f}'.format(finish - start)
        if(i % 100 == 0):
            pm.sprint(string, 0)
        else:
            pm.sprint(string, 0, savelog=False)

        # Print to screen
        if(pm.run.verbosity == 'default'):
            string = 'EXT: ' + 't = {:.5f}'.format(i*pm.sys.deltat)
            if(i % 100 == 0):
                pm.sprint(string, 1, newline=False)
            else:
                pm.sprint(string, 1, newline=False, savelog=False)
        else:
            string_one = 'residual: {:.5f}'.format(npla.norm(A_reduced*wavefunction_reduced - b_reduced))
            string_two = 'normalisation: {:.5f}'.format(norm**2)
            string_three = '--------------------------------------------------------------'
            if(i % 100 == 0):
                pm.sprint(string_one, 0)
                pm.sprint(string_two, 0)
                pm.sprint(string_three, 0)
            else:
                pm.sprint(string_one, 0, savelog=False)
                pm.sprint(string_two, 0, savelog=False)
                pm.sprint(string_three, 0, savelog=False)
  
    # Calculate the current density
    current_density = calculate_current_density(pm, density)

    return density, current_density


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

    # Construct the reduction and expansion matrices
    reduction_matrix, expansion_matrix = construct_antisymmetry_matrices(pm)

    # Construct the reduced form of the sparse matrices A and C 
    A_reduced = construct_A_reduced(pm, reduction_matrix, expansion_matrix, 0)
    C_reduced = -A_reduced + 2.0*reduction_matrix*sps.identity(pm.space.npt**2, dtype=np.float)*expansion_matrix

    # Generate the initial wavefunction
    wavefunction_reduced = initial_wavefunction(pm)

    # Propagate through imaginary time
    energy, wavefunction = solve_imaginary_time(pm, A_reduced, C_reduced, wavefunction_reduced, expansion_matrix) 
 
    # Calculate the ground-state density
    wavefunction_2D = wavefunction.reshape(pm.space.npt, pm.space.npt)
    density = calculate_density(pm, wavefunction_2D)
   
    # Save the quantities to file
    results = rs.Results()
    results.add(density,'gs_ext_den')
    results.add(energy,'gs_ext_E')
    results.add(pm.space.v_ext,'gs_ext_vxt')
    if(pm.ext.psi_gs):
        wavefunction_reduced = reduction_matrix*wavefunction
        results.add(wavefunction_reduced,'gs_ext_psi')
    if(pm.run.save):
        results.save(pm)

    # Dispose of the reduced sparse matrices
    del A_reduced
    del C_reduced
        
    # Real time 
    if(pm.run.time_dependence):

        # Array initialisations
        string = 'EXT: constructing arrays'
        pm.sprint(string, 1)
        wavefunction = wavefunction.astype(np.cfloat)

        # Construct the reduced form of the sparse matrices A and C 
        A_reduced = construct_A_reduced(pm, reduction_matrix, expansion_matrix, 1)
        C_reduced = -A_reduced + 2.0*reduction_matrix*sps.identity(pm.space.npt**2, dtype=np.cfloat)*expansion_matrix
  
        # Propagate the ground-state wavefunction through real time
        density, current_density = solve_real_time(pm, A_reduced, C_reduced, wavefunction, reduction_matrix, expansion_matrix)

        # Dispose of the reduced sparse matrices
        del A_reduced
        del C_reduced

        # Save the quantities to file
        results.add(density,'td_ext_den')
        results.add(current_density,'td_ext_cur')
        results.add(pm.space.v_ext+pm.space.v_pert,'td_ext_vxt')
        if(pm.run.save):
            results.save(pm)

    return results

