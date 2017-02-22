"""Calculates the exact ground-state electron density and energy for a system 
of two interacting electrons through solving the many-body Schrodinger 
equation. If the system is perturbed, the time-dependent electron density and 
current density are calculated. The (time-dependent) ELF can also be 
calculated.
"""


import mkl
import time
import copy
import pickle
import numpy as np
import scipy as sp
import RE_Utilities
import scipy.sparse as sps
import scipy.misc as spmisc
import scipy.special as spec
import scipy.sparse.linalg as spla
import construct_hamiltonian_coo2 as hamiltonian_coo
import construct_antisymmetry_coo2 as antisymmetry_coo
import construct_wavefunction2 as wavefunction2
import ELF 
import results as rs


def single_index(pm, j, k):
    r"""Takes every permutation of the two electron indices and creates a 
    single unique index.
    
    parameters
    ----------
    pm : object
        Parameters object
    j : integer
        1st electron index
    k : integer
        2nd electron index

    returns integer 
        Single unique index, jk
    """
    jk = k + j*pm.sys.grid

    return jk


def inverse_single_index(pm, jk):
    r"""Inverses the single_index operation. Takes the single index and returns
    the two separate electron indices.

    parameters
    ----------
    pm : object
        Parameters object
    jk : integer
        Single unique index

    returns integers
        1st electron index, j. 2nd electron index, k.
    """
    k = jk % pm.sys.grid
    j = (jk - k)/pm.sys.grid

    return j, k


def construct_antisymmetry_matrices(pm):
    r"""Constructs the reduction and expansion matrices that are used to 
    exploit the exchange antisymmetry of the wavefunction.

    parameters
    ----------
    pm : object
        Parameters object

    returns sparse matrix and sparse matrix
        Reduction matrix used to reduce the wavefunction (remove indistinct
        elements). Expansion matrix used to expand the reduced wavefunction
        (insert indistinct elements) to get back the full wavefunction.
    """
    # Number of elements in the reduced wavefunction
    coo_size = int(np.prod(range(pm.sys.grid,pm.sys.grid+2))/spmisc.factorial(
               2))

    # COOrdinate holding arrays for the reduction matrix
    coo_1 = np.zeros((coo_size), dtype=int)
    coo_2 = np.copy(coo_1)
    coo_data_1 = np.zeros((coo_size), dtype=np.float)
    
    # COOrdinate holding arrays for the expansion matrix 
    coo_3 = np.zeros((pm.sys.grid**2), dtype=int)
    coo_4 = np.copy(coo_3)
    coo_data_2 = np.zeros((pm.sys.grid**2), dtype=np.float)  

    # Populate the COOrdinate holding arrays with the coordinates and data
    coo_1, coo_2, coo_3, coo_4, coo_data_1, coo_data_2 = (antisymmetry_coo.
                       construct_antisymmetry_coo(coo_1, coo_2, coo_3, coo_4,
                       coo_data_1, coo_data_2, pm.sys.grid, coo_size))

    # Convert the holding arrays into COOrdinate sparse  matrices. Finally,
    # convert these into compressed sparse column form for efficient 
    # arithmetic.
    reduction_matrix = sps.coo_matrix((coo_data_1,(coo_1,coo_2)), shape=(
                       coo_size,pm.sys.grid**2), dtype=np.float)
    reduction_matrix = sps.csr_matrix(reduction_matrix)
    expansion_matrix = sps.coo_matrix((coo_data_2,(coo_3,coo_4)), shape=(
                       pm.sys.grid**2,coo_size), dtype=np.float)
    expansion_matrix = sps.csr_matrix(expansion_matrix)

    return reduction_matrix, expansion_matrix


def construct_A_reduced(pm, reduction_matrix, expansion_matrix, v_ext, 
                        v_coulomb, TD):
    r"""Constructs the reduced form of the sparse matrix A.

    parameters
    ----------
    pm : object
        Parameters object
    reduction_matrix : sparse matrix
        Sparse matrix used to reduce the wavefunction (remove indistinct 
        elements) by exploiting the exchange antisymmetry
    expansion_matrix : sparse matrix
        Sparse matrix used to expand the reduced wavefunction (insert 
        indistinct elements) to get back the full wavefunction
    v_ext : array_like
        1D array of the external potential, indexed as 
        v_ext[space_index] 
    v_coulomb : array_like
        1D array of the Coulomb potential, indexed as v_coulomb[space_index]
    TD : integer
        0 for the ground-state system, 1 for the time-dependent system

    returns sparse matrix
        Reduced form of the sparse matrix A, used when solving the equation Ax=b.
    """
    # Define parameter r 
    if(TD == 0):
        r = pm.ext.cdeltat/(4.0*pm.sys.deltax**2) + 0.0j
    elif(TD == 1):
        r = 0.0 + 1.0j*pm.sys.deltat/(4.0*pm.sys.deltax**2)
    
    # Construct array of the diagonal elements of the Hamiltonian that will be
    # passed to construct_hamiltonian_coo()
    hamiltonian_diagonals = construct_hamiltonian_diagonals(pm, r, v_ext,
                            v_coulomb)
 
    # Estimate the number of non-zero elements that will be in the matrix form
    # of the system's Hamiltonian, then initialize the COOrdinate sparse matrix
    # holding arrays with this shape
    coo_size = hamiltonian_coo_max_size(pm)
    coo_1 = np.zeros((coo_size), dtype=int)
    coo_2 = np.zeros((coo_size), dtype=int)
    coo_data = np.zeros((coo_size), dtype=np.cfloat)

    # Pass the holding arrays and diagonals to the Hamiltonian constructor, and
    # populate the holding arrays with the coordinates and data. Convert these
    # into a COOrdinate sparse  matrix, and finally convert this into compressed
    # sparse column form for efficient arithmetic.
    coo_1, coo_2, coo_data = hamiltonian_coo.construct_hamiltonian_coo(coo_1,
                             coo_2, coo_data, hamiltonian_diagonals, r,
                             pm.sys.grid)
    A = sps.coo_matrix((coo_data,(coo_1,coo_2)), shape=(pm.sys.grid**2,
        pm.sys.grid**2), dtype=np.cfloat)
    A = sps.csc_matrix(A)

    # Construct the reduced form of A
    A_reduced = reduction_matrix*A*expansion_matrix

    return A_reduced


def construct_hamiltonian_diagonals(pm, r, v_ext, v_coulomb):
    """Calculates the main diagonal of the Hamiltonian matrix, then stores this 
    in a Fortran contiguous array. This array is then passed to the Hamiltonian        
    constructor construct_hamiltonian_coo().

    parameters
    ----------
    pm : object
        Parameters object
    r : complex float
        Parameter in the equation Ax=b
    v_ext : array_like
        1D array of the external potential, indexed as v_ext[space_index] 
    v_coulomb : array_like
        1D array of the Coulomb potential, indexed as v_coulomb[space_index]

    returns array_like
        1D array of the Hamiltonian's main diagonal elements, indexed as 
        hamiltonian_diagonals[space_index_1_2] 
    """
    hamiltonian_diagonals = np.zeros((pm.sys.grid**2), dtype=np.cfloat, 
                            order='F')
    const = 2.0*pm.sys.deltax**2

    # Store the main diagonal elements of the Hamiltonian matrix in the optimal
    # order (i.e. the same order in which they are accessed from memory in 
    # construct_hamiltonian_coo()) to minimise the time needed to construct 
    # the full Hamiltonian matrix 
    i = 0
    for j in range(1,pm.sys.grid-1):
        for k in range(1,pm.sys.grid-1):
            hamiltonian_diagonals[i] = 1.0 + 4.0*r + const*r*potential(pm, j, 
                                       k, v_ext, v_coulomb)
            i+=1

    for j in range(0,pm.sys.grid,pm.sys.grid-1):
        for k in range(pm.sys.grid):
            hamiltonian_diagonals[i] = 1.0 + 4.0*r + const*r*potential(pm, j,
                                       k, v_ext, v_coulomb)
            i+=1

    for k in range(0,pm.sys.grid,pm.sys.grid-1):
        for j in range(1,pm.sys.grid-1):
            hamiltonian_diagonals[i] = 1.0 + 4.0*r + const*r*potential(pm, j,
                                       k, v_ext, v_coulomb)
            i+=1

    return hamiltonian_diagonals


def potential(pm, j, k, v_ext, v_coulomb):
    r"""Calculates the [j,k] element of the potential matrix.

    parameters
    ----------
    pm : object
        Parameters object
    j : integer
        1st electron index
    k : integer
        2nd electron index
    v_ext : array_like
        1D array of the external potential, indexed as v_ext[space_index] 
    v_coulomb : array_like
        1D array of the Coulomb potential, indexed as v_coulomb[space_index]

    returns float
        j, k element of potential matrix
    """
    interaction_strength = pm.sys.interaction_strength
    element = v_ext[k] + v_ext[j] + interaction_strength*v_coulomb[abs(j-k)]

    return element


def hamiltonian_coo_max_size(pm):
    """Estimates the number of non-zero elements in the Hamiltonian matrix 
    (band matrix) created by construct_hamiltonian_coo(). This accounts for
    the main diagonal (x**2) and the first off-diagonals for both electrons 
    (4x**2). This will overestimate the number of elements, resulting in an 
    array size larger than the total number of elements. These are truncated
    at the point of creation in the scipy.sparse.coo_matrix() constructor.

    parameters
    ----------
    pm : object
        Parameters object 

    returns integer
        Estimate of the number of non-zero elements in the Hamiltonian matrix
    """
    if(pm.sys.grid<3):
        print 'Warning: insufficient spatial grid points (Grid=>3).'
        return 0

    max_size = 5*pm.sys.grid**2

    return max_size


def initial_wavefunction(pm, wavefunction_reduced):
    r"""Generates the initial condition for the Crank-Nicholson imaginary 
    time propagation.

    parameters
    ----------
    pm : object
        Parameters object
    wavefunction_reduced : array_like
        1D array of the reduced wavefunction, indexed as 
        wavefunction_reduced[space_index_1_2]

    returns array_like
        1D array of the reduced wavefunction, indexed as 
        wavefunction_reduced[space_index_1_2]
    """
    eigenstate_1 = energy_eigenstate(pm,0)
    eigenstate_2 = energy_eigenstate(pm,1)
    wavefunction_reduced = wavefunction2.construct_wavefunction(eigenstate_1,
                           eigenstate_2, wavefunction_reduced, pm.sys.grid)

    return wavefunction_reduced


def energy_eigenstate(pm, n):
    r"""Calculates the nth energy eigenstate of the quantum harmonic 
    oscillator.

    parameters
    ----------
    pm : object
        Parameters object
    n : integer
        Principle quantum number

    returns array_like
        1D array of the nth eigenstate, indexed as eigenstate[space_index]
    """
    eigenstate = np.zeros(pm.sys.grid, dtype=np.cfloat, order='F')
    factorial = np.arange(0,n+1,1)
    fact = np.product(factorial[1:])
    norm = (np.sqrt(1.0/((2.0**n)*fact)))*((1.0/np.pi)**0.25)
    for j in range(pm.sys.grid):
        x = -pm.sys.xmax + j*pm.sys.deltax
        eigenstate[j] = complex(norm*(spec.hermite(n)(x))*(0.25)*np.exp(-0.5*
                        (0.25)*(x**2)), 0.0)  
 
    return eigenstate


def wavefunction_converter(pm, wavefunction):
    r"""Turns the array of compressed indices into separated indices. 

    parameters
    ----------
    pm : object
        Parameters object
    wavefunction : array_like
        1D array of the wavefunction, indexed as wavefunction[space_index_1_2]

    returns array_like
        2D array of the wavefunction, indexed as 
        wavefunction_2D[space_index_1,space_index_2]
    """
    wavefunction_2D = np.zeros((pm.sys.grid,pm.sys.grid), dtype=np.cfloat)
    for jk in range(pm.sys.grid**2):
        j, k = inverse_single_index(pm, jk)
        wavefunction_2D[j,k] = wavefunction[jk]

    return wavefunction_2D


def calculate_energy(pm, wavefunction_reduced, wavefunction_reduced_old):
    r"""Calculates the energy of the system.

    parameters
    ----------
    pm : object
        Parameters object
    wavefunction_reduced : array_like
        1D array of the reduced wavefunction at t, indexed as 
        wavefunction_reduced[space_index_1_2]
    wavefunction_reduced_old : array_like
        1D array of the reduced wavefunction at t-dt, indexed as 
        wavefunction_reduced_old[space_index_1_2]

    returns float
        Energy of the system
    """
    a = np.linalg.norm(wavefunction_reduced_old)
    b = np.linalg.norm(wavefunction_reduced)
    energy = -(np.log(b/a))/pm.ext.cdeltat

    return energy


def calculate_density(pm, wavefunction_2D):
    r"""Calculates the electron density from the two-electron wavefunction.

    parameters
    ----------
    pm : object
        Parameters object
    wavefunction : array_like
        2D array of the wavefunction, indexed as 
        wavefunction_2D[space_index_1,space_index_2]

    returns array_like
        1D array of the density, indexed as density[space_index]
    """
    mod_wavefunction_2D = np.absolute(wavefunction_2D)**2
    density = 2.0*np.sum(mod_wavefunction_2D, axis=1, dtype=np.float)*(
              pm.sys.deltax)

    return density 


def calculate_current_density(pm, density):
    r"""Calculates the current density from the time-dependent 
    (and ground-state) electron density.

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


def construct_Af(A):
    r"""Constructs the real matrix Af for parallel runs.

    parameters
    ----------
    A : sparse matrix
        Sparse matrix in the equation Ax=b

    returns sparse matrix
        Sparse matrix in the equation Ax=b for parallel runs
    """
    A1_dat, A2_dat = mkl.mkl_split(A.data,len(A.data))
    A.data = A1_dat
    A1 = copy.copy(A)
    A.data = A2_dat
    A2 = copy.copy(A)
    Af = sps.bmat([[A1,-A2],[A2,A1]]).tocsr()
 
    return Af


def solve_imaginary_time(pm, A_reduced, wavefunction_reduced, 
                         reduction_matrix, expansion_matrix):
    r"""Propagates the initial wavefunction through imaginary time using the 
    Crank-Nicholson method to find the ground-state of the system.

    parameters
    ----------
    pm : object
        Parameters object
    A_reduced : sparse matrix
        Reduced form of the sparse matrix A, used when solving the equation Ax=b.
    wavefunction_reduced : array_like
        1D array of the reduced wavefunction, indexed as
         wavefunction_reduced[space_index_1_2]
    reduction_matrix : sparse matrix
        Sparse matrix used to reduce the wavefunction (remove indistinct 
        elements) by exploiting the exchange antisymmetry
    expansion_matrix : sparse matrix
        Sparse matrix used to expand the reduced wavefunction (insert 
        indistinct elements) to get back the full wavefunction

    returns float and array_like
        Energy of the ground-state system. 1D array of the ground-state 
        wavefunction, indexed as wavefunction[space_index_1_2].
    """
    # Construct the reduced form of the sparse matrix C
    C_reduced = -A_reduced + 2.0*reduction_matrix*sps.identity(pm.sys.grid**2, 
                dtype=np.cfloat)*expansion_matrix

    # Copy the initial wavefunction
    wavefunction_reduced_old = np.copy(wavefunction_reduced)
 
    # Print to screen
    string = 'EXT: imaginary time propagation'
    pm.sprint(string, 1, newline=True)
 
    # Perform iterations
    i = 1
    while (i < pm.ext.cimax):

        # Begin timing the iteration
        start = time.time()
        string = 'complex time = {:.5f}'.format(i*pm.ext.cdeltat) 
        pm.sprint(string, 0, newline=True)

        # Save the previous time step 
        wavefunction_reduced_old[:] = wavefunction_reduced[:]

        # Construct the reduction vector of b
        if(pm.ext.par == 0):
            b_reduced = C_reduced*wavefunction_reduced
        else:
            b_reduced = mkl.mkl_mvmultiply_c(C_reduced.data,C_reduced.indptr+1,
                        C_reduced.indices+1,1,wavefunction_reduced,
                        C_reduced.shape[0],C_reduced.indices.size)

        # Solve Ax=b
        wavefunction_reduced, info = spla.cg(A_reduced,b_reduced,
                                     x0=wavefunction_reduced,tol=pm.ext.ctol)

        # Normalise the reduced wavefunction
        norm = np.linalg.norm(wavefunction_reduced)*pm.sys.deltax
        wavefunction_reduced[:] = wavefunction_reduced[:]/norm
           
        # Stop timing the iteration
        finish = time.time()
        string = 'time to complete step: {:.5f}'.format(finish - start)
        pm.sprint(string, 0, newline=True)

        # Test for convergence
        wavefunction_convergence = np.linalg.norm(wavefunction_reduced_old -
                                   wavefunction_reduced)
        string = 'wavefunction convergence: ' + str(wavefunction_convergence)
        pm.sprint(string, 0, newline=True) 
        if(pm.run.verbosity=='default'):
            string = 'EXT: ' + 't = {:.5f}'.format(i*pm.ext.cdeltat) + \
                     ', convergence = ' + str(wavefunction_convergence)
            pm.sprint(string, 1, newline=False)
        if(wavefunction_convergence < pm.ext.ctol*10.0):
            i = pm.ext.cimax
            pm.sprint('', 1, newline=True)
            string = 'EXT: ground-state converged' 
            pm.sprint(string, 1, newline=True)
        string = '--------------------------------------------------------' + \
                 '----------'
        pm.sprint(string, 0, newline=True)

        # Iterate
        i += 1

    # Calculate the ground-state energy
    wavefunction_reduced[:] = norm*wavefunction_reduced[:]
    energy = calculate_energy(pm, wavefunction_reduced, 
             wavefunction_reduced_old)
    string = 'EXT: ground-state energy = {:.5f}'.format(energy)
    pm.sprint(string, 1, newline=True)

    # Expand the ground-state wavefunction and normalise
    wavefunction = expansion_matrix*wavefunction_reduced
    norm = np.linalg.norm(wavefunction)*pm.sys.deltax
    wavefunction[:] = wavefunction[:]/norm 

    return energy, wavefunction


def solve_real_time(pm, A_reduced, wavefunction, reduction_matrix,
                    expansion_matrix):
    r"""Propagates the ground-state wavefunction through real time using the 
    Crank-Nicholson method to find the time-evolution of the perturbed system.

    parameters
    ----------
    pm : object
        Parameters object
    A_reduced : sparse matrix
        Reduced form of the sparse matrix A, used when solving the equation Ax=b.
    wavefunction : array_like
        1D array of the ground-state wavefunction, indexed as 
        wavefunction[space_index_1_2]
    reduction_matrix : sparse matrix
        Sparse matrix used to reduce the wavefunction (remove indistinct 
        elements) by exploiting the exchange antisymmetry
    expansion_matrix : sparse matrix
        Sparse matrix used to expand the reduced wavefunction (insert 
        indistinct elements) to get back the full wavefunction

    returns array_like and array_like
        2D array of the time-dependent density, indexed as 
        density[time_index,space_index]. 2D array of the current density, 
        indexed as current_density[time_index,space_index].
    """
    # Construct the matrix Af for a parallel run 
    if(pm.ext.par == 1):
        Af = ConstructAf(A_reduced)

    # Construct the reduction matrix of C
    C_reduced = -A_reduced + 2.0*reduction_matrix*sps.identity(pm.sys.grid**2, 
                dtype=np.cfloat)*expansion_matrix

    # Array initialisations
    density = np.zeros((pm.sys.imax+1,pm.sys.grid), dtype=np.float)
    if(pm.ext.ELF_TD == 1):
        elf = np.copy(density)
    else:
        elf = 0 

    # Save ground state
    wavefunction_2D = wavefunction_converter(pm, wavefunction)
    density[0,:] = calculate_density(pm, wavefunction_2D)
   
    # Reduce the wavefunction
    wavefunction_reduced = reduction_matrix*wavefunction

    # Print to screen
    string = 'EXT: real time propagation'
    pm.sprint(string, 1, newline=True)

    # Perform iterations
    for i in range(pm.sys.imax):

        # Begin timing the iteration
        start = time.time()
        string = 'real time = {:.5f}'.format((i+1)*pm.sys.deltat) + '/' + \
                 '{:.5f}'.format((pm.sys.imax)*pm.sys.deltat)
        pm.sprint(string, 0, newline=True)

        # Construct the vector b and its reduction vector
        if(pm.run.verbosity == 'high'):
            b = C*wavefunction[:]
        if(pm.ext.par == 0):
            b_reduced = C_reduced*wavefunction_reduced
        else:
            b_reduced = mkl.mkl_mvmultiply_c(C_reduced.data,C_reduced.indptr+1,
                        C_reduced.indices+1,1,wavefunction_reduced,
                        C_reduced.shape[0],C_reduced.indices.size)

        # Solve Ax=b
        if(pm.ext.par == 0):
            wavefunction_reduced, info = spla.cg(A_reduced,b_reduced,
                                         x0=wavefunction_reduced,tol=pm.ext.rtol)
        else:
            b1, b2 = mkl.mkl_split(b_reduced,len(b_reduced))
            bf = np.append(b1,b2)
            if(i == 0):
                xf = bf
            xf = mkl.mkl_isolve(Af.data,Af.indptr+1,Af.indices+1,1,bf,xf,
                 Af.shape[0],Af.indices.size)
            x1, x2 = np.split(xf,2)
            wavefunction_reduced = mkl.mkl_comb(x1,x2,len(x1))

        # Expand the wavefunction
        wavefunction = expansion_matrix*wavefunction_reduced

        # Calculate the density (and ELF)
        wavefunction_2D = wavefunction_converter(pm, wavefunction)
        density[i+1,:] = calculate_density(pm, wavefunction_2D)
        if(pm.ext.ELF_TD == 1):
            elf[i+1,:] = ELF.main(pm, wavefunction_2D)
  
        # Stop timing the iteration
        finish = time.time()
        string = 'time to complete step: {:.5f}'.format(finish - start)
        pm.sprint(string, 0, newline=True)

        # Print to screen
        if(pm.run.verbosity == 'default'):
            string = 'EXT: ' + 't = {:.5f}'.format((i+1)*pm.sys.deltat)
            pm.sprint(string, 1, newline=False)
        else:
            string = 'residual: {:.5f}'.format(np.linalg.norm(A*wavefunction
                     - b))
            pm.sprint(string, 0, newline=True)
            norm = np.linalg.norm(wavefunction)*pm.sys.deltax
            string = 'normalisation: {:.5f}'.format(norm**2)
            pm.sprint(string, 0, newline=True)
            string = '----------------------------------------------------' + \
                     '----------'
            pm.sprint(string, 0, newline=True)
  
    # Calculate the current density
    current_density = calculate_current_density(pm, density)

    if(pm.ext.ELF_TD == 1):
        return density[1:], current_density, elf[1:]
    else:
        return density[1:], current_density, elf


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

    # Array initialisations 
    string = 'EXT: constructing arrays'
    pm.sprint(string, 1, newline=True)
    wavefunction = np.zeros(pm.sys.grid**2, dtype=np.cfloat)
    x_points = np.linspace(-pm.sys.xmax,pm.sys.xmax,pm.sys.grid)
    v_ext = pm.sys.v_ext(x_points)
    v_pert = pm.sys.v_pert(x_points)
    x_points_tmp = np.linspace(0.0,2.0*pm.sys.xmax,pm.sys.grid)
    v_coulomb = 1.0/(pm.sys.acon + x_points_tmp)

    # Construct the reduction and expansion matrices
    reduction_matrix, expansion_matrix = construct_antisymmetry_matrices(pm)

    # Construct the reduced form of the sparse matrix A 
    A_reduced = construct_A_reduced(pm, reduction_matrix, expansion_matrix,
                v_ext, v_coulomb, 0)

    # Generate the initial wavefunction
    wavefunction_reduced = np.zeros(reduction_matrix.shape[0], dtype=np.cfloat)
    wavefunction_reduced = initial_wavefunction(pm, wavefunction_reduced)
    
    # Propagate through imaginary time
    energy, wavefunction = solve_imaginary_time(pm, A_reduced,
                           wavefunction_reduced, reduction_matrix,
                           expansion_matrix) 
  
    # Dispose of the reduced matrix
    del A_reduced

    # Calculate ground-state density (and ELF)
    wavefunction_2D = wavefunction_converter(pm, wavefunction)
    density = calculate_density(pm, wavefunction_2D)
    if(pm.ext.ELF_GS == 1):
        elf = ELF.main(pm, wavefunction_2D)
   
    # Save ground-state density, energy and external potential (and ELF)
    results = rs.Results()
    results.add(density,'gs_ext_den')
    results.add(energy,'gs_ext_E')
    results.add(v_ext,'gs_ext_vxt')
    if(pm.ext.ELF_GS == 1):
        results.add(elf,'gs_ext_elf')
    if(pm.run.save):
        results.save(pm)
        
    # Propagate through real time
    if(pm.run.time_dependence == True):
        string = 'EXT: constructing arrays'
        pm.sprint(string,1,newline=True)
        v_ext += v_pert
        A_reduced = construct_A_reduced(pm, reduction_matrix, expansion_matrix,
                    v_ext, v_coulomb, 1)
        density, current_density, elf = solve_real_time(pm, A_reduced,
                                        wavefunction, reduction_matrix,
                                        expansion_matrix)
        del A_reduced

        # Save time-dependent density, current density and external potential 
        # (and ELF)
        results.add(density,'td_ext_den')
        results.add(current_density,'td_ext_cur')
        results.add(v_ext,'td_ext_vxt')
        if(pm.ext.ELF_TD == 1):
            results.add(elf,'td_ext_elf')
        if(pm.run.save):
            results.save(pm)

    return results
