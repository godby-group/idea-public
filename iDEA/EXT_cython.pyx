"""Contains the cython modules that are called within EXT2 and EXT3. Cython is 
used for operations that are very expensive to do in Python, and performance 
speeds are close to C.
"""
cimport numpy as np
import numpy as np

#cython: boundscheck=False, wraparound=False, nonecheck=False


def wavefunction_two(double[:] eigenstate_1, double[:] eigenstate_2):           
    r"""Constructs the two-electron initial wavefunction in reduced form from
    two single-particle eigenstates.

    parameters
    ----------
    eigenstate_1 : array_like
        1D array of the 1st single-particle eigenstate, indexed as 
        eigenstate_1[space_index]
    eigenstate_2 : array_like
        1D array of the 2nd single-particle eigenstate, indexed as 
        eigenstate_2[space_index]
    grid : integer
        Number of spatial grid points in the system

    returns array_like
        1D array of the reduced wavefunction, indexed as 
        wavefunction_reduced[space_index_1_2]
    """
    # Variable declarations
    cdef int i, j, k
    cdef int grid = eigenstate_1.shape[0]
    cdef double normalisation
    cdef double[:] wavefunction_reduced = np.zeros(int(grid*(grid+1)/2), dtype=np.float)

    # Normalisation factor
    normalisation = 1.0/np.sqrt(2.0)

    # Loop over each element of the reduced wavefunction
    i = 0
    for j in range(grid):
        for k in range(j+1):
           
            # Calculate permutation from Slater determinant 
            wavefunction_reduced[i] = normalisation*(eigenstate_1[j]*eigenstate_2[k] - eigenstate_1[k]*eigenstate_2[j])
             
            # Increase count
            i += 1

    return np.asarray(wavefunction_reduced, dtype=np.float)


def wavefunction_three(double[:] eigenstate_1, double[:] eigenstate_2, double[:] eigenstate_3):           
    r"""Constructs the initial three-electron wavefunction in reduced form from
    three single-particle eigenstates.

    parameters
    ----------
    eigenstate_1 : array_like
        1D array of the 1st single-particle eigenstate, indexed as 
        eigenstate_1[space_index]
    eigenstate_2 : array_like
        1D array of the 2nd single-particle eigenstate, indexed as 
        eigenstate_2[space_index]
    eigenstate_3 : array_like
        1D array of the 3rd single-particle eigenstate, indexed as 
        eigenstate_3[space_index]
    grid : integer
        Number of spatial grid points in the system

    returns array_like
        1D array of the reduced wavefunction, indexed as 
        wavefunction_reduced[space_index_1_2_3]
    """
    # Variable declarations
    cdef int i, j, k, l
    cdef int grid = eigenstate_1.shape[0]
    cdef double normalisation, perm_1, perm_2, perm_3
    cdef double[:] wavefunction_reduced = np.zeros(int(grid*(grid+1)*(grid+2)/6), dtype=np.float)

    # Normalisation factor
    normalisation = 1.0/np.sqrt(6.0)

    # Loop over each element of the reduced wavefunction
    i = 0
    for j in range(grid):
        for k in range(j+1):
            for l in range(k+1):
           
                # Calculate permutations from Slater determinant 
                perm_1 = eigenstate_1[j]*(eigenstate_2[k]*eigenstate_3[l] - eigenstate_3[k]*eigenstate_2[l])
                perm_2 = eigenstate_2[j]*(eigenstate_3[k]*eigenstate_1[l] - eigenstate_1[k]*eigenstate_3[l])
                perm_3 = eigenstate_3[j]*(eigenstate_1[k]*eigenstate_2[l] - eigenstate_2[k]*eigenstate_1[l])
                wavefunction_reduced[i] = normalisation*(perm_1 + perm_2 + perm_3)
    
                # Increase count
                i += 1

    return np.asarray(wavefunction_reduced, dtype=np.float)


def reduction_two(np.int64_t[:] coo_1, np.int64_t[:] coo_2, int grid):
    r"""Calculates the coordinates and data of the non-zero elements of the 
    two-electron reduction matrix that is used to exploit the exchange exchange 
    antisymmetry of the wavefunction.

    parameters
    ----------
    coo_1 : array_like
        1D COOrdinate holding array for the non-zero elements of the reduction 
        matrix
    coo_2 : array_like
        1D COOrdinate holding array for the non-zero elements of the reduction  
        matrix
    grid : integer
        Number of spatial grid points in the system

    returns array_like and array_like 
        Populated 1D COOrdinate holding arrays for the non-zero elements of the
        reduction matrix
    """
    # Variable declarations
    cdef int i, j, k, jk

    # Loop over each non-zero element of the reduction matrix
    i = 0
    for j in range(grid):
        for k in range(j+1):
        
            # Calculate element indices
            coo_1[i] = i
            coo_2[i] = single_index_two(j, k, grid)

            # Increase count
            i += 1

    return coo_1, coo_2


def reduction_three(np.int64_t[:] coo_1, np.int64_t[:] coo_2, int grid):
    r"""Calculates the coordinates and data of the non-zero elements of the 
    three-electron reduction matrix that is used to exploit the exchange 
    exchange antisymmetry of the wavefunction.

    parameters
    ----------
    coo_1 : array_like
        1D COOrdinate holding array for the non-zero elements of the reduction 
        matrix
    coo_2 : array_like
        1D COOrdinate holding array for the non-zero elements of the reduction  
        matrix
    grid : integer
        Number of spatial grid points in the system

    returns array_like and array_like 
        Populated 1D COOrdinate holding arrays for the non-zero elements of the
        reduction matrix
    """
    # Variable declarations
    cdef int i, j, k, l, jkl

    # Loop over each non-zero element of the reduction matrix
    i = 0
    for j in range(grid):
        for k in range(j+1):
            for l in range(k+1):
        
                # Calculate element indices
                coo_1[i] = i
                coo_2[i] = single_index_three(j, k, l, grid)

                # Increase count
                i += 1

    return coo_1, coo_2


def expansion_two(np.int64_t[:] coo_1, np.int64_t[:] coo_2, double[:] coo_data, int grid):
    r"""Calculates the coordinates and data of the non-zero elements of the 
    two-electron expansion matrix that is used to exploit the exchange exchange 
    antisymmetry of the wavefunction.

    parameters
    ----------
    coo_1 : array_like
        1D COOrdinate holding array for the non-zero elements of the expansion 
        matrix
    coo_2 : array_like
        1D COOrdinate holding array for the non-zero elements of the expansion 
        matrix
    coo_data : array_like
        1D array of the non-zero elements of the expansion matrix
    grid : integer
        Number of spatial grid points in the system

    returns array_like and array_like and array_like
        Populated 1D COOrdinate holding arrays and 1D data holding array for the 
        non-zero elements of the expansion matrix
    """
    # Variable declarations
    cdef int i_plus, i_minus, j, k, jk, kj, element

    # Loop over each non-zero element of the expansion matrix
    i_plus = 0
    i_minus = 0
    for j in range(grid):
        for k in range(j+1):
        
            # Calculate positive element indices
            coo_1[i_plus] = single_index_two(j, k, grid)
            coo_2[i_plus] = i_plus
            coo_data[i_plus] = 1.0

            # Calculate negative element indices
            if(j != k):
                element = grid**2-1-i_minus
                coo_1[element] = single_index_two(k, j, grid)
                coo_2[element] = i_plus
                coo_data[element] = -1.0
                i_minus += 1

            # Increase count
            i_plus += 1

    return np.asarray(coo_1, dtype=int), np.asarray(coo_2, dtype=int), np.asarray(coo_data, dtype=np.float)


def expansion_three(np.int64_t[:] coo_1, np.int64_t[:] coo_2, double[:] coo_data, int grid):
    r"""Calculates the coordinates and data of the non-zero elements of the 
    three-electron expansion matrix that is used to exploit the exchange 
    exchange antisymmetry of the wavefunction.

    parameters
    ----------
    coo_1 : array_like
        1D COOrdinate holding array for the non-zero elements of the expansion 
        matrix
    coo_2 : array_like
        1D COOrdinate holding array for the non-zero elements of the expansion 
        matrix
    coo_data : array_like
        1D array of the non-zero elements of the expansion matrix
    grid : integer
        Number of spatial grid points in the system

    returns array_like and array_like and array_like
        Populated 1D COOrdinate holding arrays and 1D data holding array for the 
        non-zero elements of the expansion matrix
    """
    # Variable declarations
    cdef int i_plus, i_minus, j, k, l, jkl, jlk, klj, kjl, ljk, lkj, element

    # Loop over each non-zero element of the expansion matrix
    i_plus = 0
    i_minus = 0
    for j in range(grid):
        for k in range(j+1):
            for l in range(k+1):
        
                # Calculate positive element indices
                coo_1[i_plus] = single_index_three(j, k, l, grid)
                coo_2[i_plus] = i_plus
                coo_data[i_plus] = 1.0

                # Calculate negative element indices
                if(j != k and k == l):
                    element = grid**3-1-i_minus
                    coo_1[element] = single_index_three(k, j, l, grid)
                    coo_2[element] = i_plus
                    coo_data[element] = -1.0
                    i_minus += 1
                    
                    element -= 1
                    coo_1[element] = single_index_three(l, k, j, grid)
                    coo_2[element] = i_plus
                    coo_data[element] = -1.0
                    i_minus += 1
     
                if(j == k and k != l):
                    element = grid**3-1-i_minus
                    coo_1[element] = single_index_three(j, l, k, grid)
                    coo_2[element] = i_plus
                    coo_data[element] = -1.0
                    i_minus += 1
 
                    element -= 1
                    coo_1[element] = single_index_three(l, k, j, grid)
                    coo_2[element] = i_plus
                    coo_data[element] = -1.0
                    i_minus += 1

                if(j != k and k != l): 
                    element = grid**3-1-i_minus
                    coo_1[element] = single_index_three(j, l, k, grid)
                    coo_2[element] = i_plus
                    coo_data[element] = -1.0
                    i_minus += 1
        
                    element -= 1
                    coo_1[element] = single_index_three(k, j, l, grid)
                    coo_2[element] = i_plus
                    coo_data[element] = -1.0
                    i_minus += 1

                    element -= 1 
                    coo_1[element] = single_index_three(l, k, j, grid)
                    coo_2[element] = i_plus
                    coo_data[element] = -1.0
                    i_minus += 1
 
                    element -= 1                    
                    coo_1[element] = single_index_three(k, l, j, grid)
                    coo_2[element] = i_plus
                    coo_data[element] = 1.0
                    i_minus += 1

                    element -= 1 
                    coo_1[element] = single_index_three(l, j, k, grid)
                    coo_2[element] = i_plus
                    coo_data[element] = 1.0
                    i_minus += 1

                # Increase count
                i_plus += 1

    return np.asarray(coo_1, dtype=int), np.asarray(coo_2, dtype=int), np.asarray(coo_data, dtype=np.float)


def hamiltonian_two(object pm, np.int64_t[:] coo_1, np.int64_t[:] coo_2, double [:] coo_data, int td):
    r"""Calculates the coordinates and data of the non-zero elements of the 
    two-electron Hamiltonian matrix.

    parameters
    ----------
    pm : object
        Parameters object
    coo_1 : array_like
        1D COOrdinate holding array for the non-zero elements of the Hamiltonian
        matrix
    coo_2 : array_like
        1D COOrdinate holding array for the non-zero elements of the Hamiltonian
        matrix
    coo_data : array_like
        1D array of the non-zero elements of the Hamiltonian matrix 
    td : integer
        0 for imaginary time, 1 for real time
  
    returns array_like and array_like and array_like
        Populated 1D COOrdinate holding arrays and 1D data holding array for the 
        non-zero elements of the Hamiltonian matrix
    """
    # Variable declarations
    cdef int i, j, k, jk, band
    cdef int grid = pm.space.npt
    cdef double[:] ke_bands = -0.5*pm.space.second_derivative_band
    cdef int bandwidth = ke_bands.shape[0]
    cdef double[:] v_ext = pm.space.v_ext
    cdef double[:,:] v_int = pm.space.v_int

    # Add the perturbing potential if there is no imaginary part
    if(td == 1 and pm.sys.im == 0):
        v_ext += pm.space.v_pert

    # Loop over each non-zero element of the Hamiltonian matrix  
    i = 0
    for j in range(bandwidth-1, grid-bandwidth+1):
        for k in range(bandwidth-1, grid-bandwidth+1):
 
          # Main diagonal
            jk = single_index_two(j, k, grid)
            coo_1[i] = jk
            coo_2[i] = jk
            coo_data[i] = 2.0*ke_bands[0] + v_ext[j] + v_ext[k] + v_int[j,k]
            i += 1

          # Off-diagonals
            for band in range(1, bandwidth):
                coo_1[i] = single_index_two(j+band, k, grid)
                coo_2[i] = jk
                coo_data[i] = ke_bands[band]
                i += 1

                coo_1[i] = single_index_two(j-band, k, grid)
                coo_2[i] = jk
                coo_data[i] = ke_bands[band]
                i += 1

                coo_1[i] = single_index_two(j, k+band, grid)
                coo_2[i] = jk   
                coo_data[i] = ke_bands[band]
                i += 1

                coo_1[i] = single_index_two(j, k-band, grid)
                coo_2[i] = jk
                coo_data[i] = ke_bands[band]
                i += 1


    for j in range(bandwidth-1):
        for k in range(grid):

            # Main diagonal
            jk = single_index_two(j, k, grid)
            coo_1[i] = jk
            coo_2[i] = jk
            coo_data[i] = 2.0*ke_bands[0] + v_ext[j] + v_ext[k] + v_int[j,k]
            i += 1
   
            # Off-diagonals
            for band in range(1, bandwidth):
                if(j < grid-band):
                    coo_1[i] = single_index_two(j+band, k, grid)
                    coo_2[i] = jk
                    coo_data[i] = ke_bands[band]
                    i += 1

                if(j > band-1):
                    coo_1[i] = single_index_two(j-band, k, grid)
                    coo_2[i] = jk
                    coo_data[i] = ke_bands[band]
                    i += 1
 
                if(k < grid-band):
                    coo_1[i] = single_index_two(j, k+band, grid)
                    coo_2[i] = jk
                    coo_data[i] = ke_bands[band]
                    i += 1

                if(k > band-1):
                    coo_1[i] = single_index_two(j, k-band, grid)
                    coo_2[i] = jk
                    coo_data[i] = ke_bands[band]
                    i += 1


    for j in range(grid-bandwidth+1, grid):
        for k in range(grid):

            # Main diagonal
            jk = single_index_two(j, k, grid)
            coo_1[i] = jk
            coo_2[i] = jk
            coo_data[i] = 2.0*ke_bands[0] + v_ext[j] + v_ext[k] + v_int[j,k]
            i += 1

            # Off-diagonals
            for band in range(1, bandwidth):
                if(j < grid-band):
                    coo_1[i] = single_index_two(j+band, k, grid)
                    coo_2[i] = jk
                    coo_data[i] = ke_bands[band]
                    i += 1

                if(j > band-1):
                    coo_1[i] = single_index_two(j-band, k, grid)
                    coo_2[i] = jk
                    coo_data[i] = ke_bands[band]
                    i += 1
 
                if(k < grid-band):
                    coo_1[i] = single_index_two(j, k+band, grid)
                    coo_2[i] = jk
                    coo_data[i] = ke_bands[band]
                    i += 1

                if(k > band-1):
                    coo_1[i] = single_index_two(j, k-band, grid)
                    coo_2[i] = jk
                    coo_data[i] = ke_bands[band]
                    i += 1


    for k in range(bandwidth-1):
        for j in range(bandwidth-1, grid-bandwidth+1):

            # Main diagonal
            jk = single_index_two(j, k, grid)
            coo_1[i] = jk
            coo_2[i] = jk
            coo_data[i] = 2.0*ke_bands[0] + v_ext[j] + v_ext[k] + v_int[j,k]
            i += 1

            # Off-diagonals
            for band in range(1, bandwidth):
                if(j < grid-band):
                    coo_1[i] = single_index_two(j+band, k, grid)
                    coo_2[i] = jk
                    coo_data[i] = ke_bands[band]
                    i += 1

                if(j > band-1):
                    coo_1[i] = single_index_two(j-band, k, grid)
                    coo_2[i] = jk
                    coo_data[i] = ke_bands[band]
                    i += 1
 
                if(k < grid-band):
                    coo_1[i] = single_index_two(j, k+band, grid)
                    coo_2[i] = jk
                    coo_data[i] = ke_bands[band]
                    i += 1

                if(k > band-1):
                    coo_1[i] = single_index_two(j, k-band, grid)
                    coo_2[i] = jk
                    coo_data[i] = ke_bands[band]
                    i += 1


    for k in range(grid-bandwidth+1, grid):
        for j in range(bandwidth-1, grid-bandwidth+1):

            # Main diagonal
            jk = single_index_two(j, k, grid)
            coo_1[i] = jk
            coo_2[i] = jk
            coo_data[i] = 2.0*ke_bands[0] + v_ext[j] + v_ext[k] + v_int[j,k]
            i += 1

            # Off-diagonals
            for band in range(1, bandwidth):
                if(j < grid-band):
                    coo_1[i] = single_index_two(j+band, k, grid)
                    coo_2[i] = jk
                    coo_data[i] = ke_bands[band]
                    i += 1

                if(j > band-1):
                    coo_1[i] = single_index_two(j-band, k, grid)
                    coo_2[i] = jk
                    coo_data[i] = ke_bands[band]
                    i += 1
 
                if(k < grid-band):
                    coo_1[i] = single_index_two(j, k+band, grid)
                    coo_2[i] = jk
                    coo_data[i] = ke_bands[band]
                    i += 1

                if(k > band-1):
                    coo_1[i] = single_index_two(j, k-band, grid)
                    coo_2[i] = jk
                    coo_data[i] = ke_bands[band]
                    i += 1


    return np.asarray(coo_1, dtype=int), np.asarray(coo_2, dtype=int), np.asarray(coo_data, dtype=np.float)


def hamiltonian_three(object pm, np.int64_t[:] coo_1, np.int64_t[:] coo_2, double [:] coo_data, int td):
    r"""Calculates the coordinates and data of the non-zero elements of the 
    three-electron Hamiltonian matrix.

    parameters
    ----------
    pm : object
        Parameters object
    coo_1 : array_like
        1D COOrdinate holding array for the non-zero elements of the Hamiltonian
        matrix
    coo_2 : array_like
        1D COOrdinate holding array for the non-zero elements of the Hamiltonian
        matrix
    coo_data : array_like
        1D array of the non-zero elements of the Hamiltonian matrix 
    td : integer
        0 for imaginary time, 1 for real time
  
    returns array_like and array_like and array_like
        Populated 1D COOrdinate holding arrays and 1D data holding array for the 
        non-zero elements of the Hamiltonian matrix
    """
    # Variable declarations
    cdef int i, j, k, l, jkl, band
    cdef int grid = pm.space.npt
    cdef double[:] ke_bands = -0.5*pm.space.second_derivative_band
    cdef int bandwidth = ke_bands.shape[0]
    cdef double[:] v_ext = pm.space.v_ext
    cdef double[:,:] v_int = pm.space.v_int

    # Add the perturbing potential if there is no imaginary part
    if(td == 1 and pm.sys.im == 0):
        v_ext += pm.space.v_pert

    # Loop over each non-zero element of the Hamiltonian matrix  
    i = 0
    for j in range(bandwidth-1, grid-bandwidth+1):
        for k in range(bandwidth-1, grid-bandwidth+1):
            for l in range(bandwidth-1, grid-bandwidth+1):

                # Main diagonal
                jkl = single_index_three(j, k, l, grid)
                coo_1[i] = jkl
                coo_2[i] = jkl
                coo_data[i] = 3.0*ke_bands[0] + v_ext[j] + v_ext[k] + v_ext[l] + v_int[j,k] + v_int[j,l] + v_int[k,l]
                i += 1

                # Off-diagonals
                for band in range(1, bandwidth):
                    coo_1[i] = single_index_three(j+band, k, l, grid)
                    coo_2[i] = jkl
                    coo_data[i] = ke_bands[band]
                    i += 1

                    coo_1[i] = single_index_three(j-band, k, l, grid)
                    coo_2[i] = jkl
                    coo_data[i] = ke_bands[band]
                    i += 1

                    coo_1[i] = single_index_three(j, k+band, l, grid)
                    coo_2[i] = jkl
                    coo_data[i] = ke_bands[band]
                    i += 1

                    coo_1[i] = single_index_three(j, k-band, l, grid)
                    coo_2[i] = jkl
                    coo_data[i] = ke_bands[band]
                    i += 1

                    coo_1[i] = single_index_three(j, k, l+band, grid)
                    coo_2[i] = jkl
                    coo_data[i] = ke_bands[band]
                    i += 1

                    coo_1[i] = single_index_three(j, k, l-band, grid)
                    coo_2[i] = jkl
                    coo_data[i] = ke_bands[band]
                    i += 1


    for j in range(bandwidth-1):
        for k in range(grid):
            for l in range(grid):
  
                # Main diagonal
                jkl = single_index_three(j, k, l, grid)
                coo_1[i] = jkl
                coo_2[i] = jkl
                coo_data[i] = 3.0*ke_bands[0] + v_ext[j] + v_ext[k] + v_ext[l] + v_int[j,k] + v_int[j,l] + v_int[k,l]
                i += 1

                # Off-diagonals
                for band in range(1, bandwidth):
                    if(j < grid-band):
                        coo_1[i] = single_index_three(j+band, k, l, grid)
                        coo_2[i] = jkl
                        coo_data[i] = ke_bands[band]
                        i += 1

                    if(j > band-1):
                        coo_1[i] = single_index_three(j-band, k, l, grid)
                        coo_2[i] = jkl
                        coo_data[i] = ke_bands[band]
                        i += 1
 
                    if(k < grid-band):
                        coo_1[i] = single_index_three(j, k+band, l, grid)
                        coo_2[i] = jkl
                        coo_data[i] = ke_bands[band]
                        i += 1

                    if(k > band-1):
                        coo_1[i] = single_index_three(j, k-band, l, grid)
                        coo_2[i] = jkl
                        coo_data[i] = ke_bands[band]
                        i += 1

                    if(l < grid-band):
                        coo_1[i] = single_index_three(j, k, l+band, grid)
                        coo_2[i] = jkl
                        coo_data[i] = ke_bands[band]
                        i += 1

                    if(l > band-1):
                        coo_1[i] = single_index_three(j, k, l-band, grid)
                        coo_2[i] = jkl
                        coo_data[i] = ke_bands[band]
                        i += 1


    for j in range(grid-bandwidth+1, grid):
        for k in range(grid):
            for l in range(grid):

                # Main diagonal
                jkl = single_index_three(j, k, l, grid)
                coo_1[i] = jkl
                coo_2[i] = jkl
                coo_data[i] = 3.0*ke_bands[0] + v_ext[j] + v_ext[k] + v_ext[l] + v_int[j,k] + v_int[j,l] + v_int[k,l]
                i += 1

                # Off-diagonals
                for band in range(1, bandwidth):
                    if(j < grid-band):
                        coo_1[i] = single_index_three(j+band, k, l, grid)
                        coo_2[i] = jkl
                        coo_data[i] = ke_bands[band]
                        i += 1

                    if(j > band-1):
                        coo_1[i] = single_index_three(j-band, k, l, grid)
                        coo_2[i] = jkl
                        coo_data[i] = ke_bands[band]
                        i += 1
 
                    if(k < grid-band):
                        coo_1[i] = single_index_three(j, k+band, l, grid)
                        coo_2[i] = jkl
                        coo_data[i] = ke_bands[band]
                        i += 1

                    if(k > band-1):
                        coo_1[i] = single_index_three(j, k-band, l, grid)
                        coo_2[i] = jkl
                        coo_data[i] = ke_bands[band]
                        i += 1

                    if(l < grid-band):
                        coo_1[i] = single_index_three(j, k, l+band, grid)
                        coo_2[i] = jkl
                        coo_data[i] = ke_bands[band]
                        i += 1

                    if(l > band-1):
                        coo_1[i] = single_index_three(j, k, l-band, grid)
                        coo_2[i] = jkl
                        coo_data[i] = ke_bands[band]
                        i += 1


    for k in range(bandwidth-1):
        for j in range(bandwidth-1, grid-bandwidth+1):
            for l in range(grid):

                # Main diagonal
                jkl = single_index_three(j, k, l, grid)
                coo_1[i] = jkl
                coo_2[i] = jkl
                coo_data[i] = 3.0*ke_bands[0] + v_ext[j] + v_ext[k] + v_ext[l] + v_int[j,k] + v_int[j,l] + v_int[k,l]
                i += 1

                # Off-diagonals
                for band in range(1, bandwidth):
                    if(j < grid-band):
                        coo_1[i] = single_index_three(j+band, k, l, grid)
                        coo_2[i] = jkl
                        coo_data[i] = ke_bands[band]
                        i += 1

                    if(j > band-1):
                        coo_1[i] = single_index_three(j-band, k, l, grid)
                        coo_2[i] = jkl
                        coo_data[i] = ke_bands[band]
                        i += 1
 
                    if(k < grid-band):
                        coo_1[i] = single_index_three(j, k+band, l, grid)
                        coo_2[i] = jkl
                        coo_data[i] = ke_bands[band]
                        i += 1

                    if(k > band-1):
                        coo_1[i] = single_index_three(j, k-band, l, grid)
                        coo_2[i] = jkl
                        coo_data[i] = ke_bands[band]
                        i += 1

                    if(l < grid-band):
                        coo_1[i] = single_index_three(j, k, l+band, grid)
                        coo_2[i] = jkl
                        coo_data[i] = ke_bands[band]
                        i += 1

                    if(l > band-1):
                        coo_1[i] = single_index_three(j, k, l-band, grid)
                        coo_2[i] = jkl
                        coo_data[i] = ke_bands[band]
                        i += 1


    for k in range(grid-bandwidth+1, grid):
        for j in range(bandwidth-1, grid-bandwidth+1):
            for l in range(grid):

                # Main diagonal
                jkl = single_index_three(j, k, l, grid)
                coo_1[i] = jkl
                coo_2[i] = jkl
                coo_data[i] = 3.0*ke_bands[0] + v_ext[j] + v_ext[k] + v_ext[l] + v_int[j,k] + v_int[j,l] + v_int[k,l]
                i += 1

                # Off-diagonals
                for band in range(1, bandwidth):
                    if(j < grid-band):
                        coo_1[i] = single_index_three(j+band, k, l, grid)
                        coo_2[i] = jkl
                        coo_data[i] = ke_bands[band]
                        i += 1

                    if(j > band-1):
                        coo_1[i] = single_index_three(j-band, k, l, grid)
                        coo_2[i] = jkl
                        coo_data[i] = ke_bands[band]
                        i += 1
 
                    if(k < grid-band):
                        coo_1[i] = single_index_three(j, k+band, l, grid)
                        coo_2[i] = jkl
                        coo_data[i] = ke_bands[band]
                        i += 1

                    if(k > band-1):
                        coo_1[i] = single_index_three(j, k-band, l, grid)
                        coo_2[i] = jkl
                        coo_data[i] = ke_bands[band]
                        i += 1

                    if(l < grid-band):
                        coo_1[i] = single_index_three(j, k, l+band, grid)
                        coo_2[i] = jkl
                        coo_data[i] = ke_bands[band]
                        i += 1

                    if(l > band-1):
                        coo_1[i] = single_index_three(j, k, l-band, grid)
                        coo_2[i] = jkl
                        coo_data[i] = ke_bands[band]
                        i += 1


    for l in range(bandwidth-1):
        for j in range(bandwidth-1, grid-bandwidth+1):
            for k in range(bandwidth-1, grid-bandwidth+1):

                # Main diagonal
                jkl = single_index_three(j, k, l, grid)
                coo_1[i] = jkl
                coo_2[i] = jkl
                coo_data[i] = 3.0*ke_bands[0] + v_ext[j] + v_ext[k] + v_ext[l] + v_int[j,k] + v_int[j,l] + v_int[k,l]
                i += 1

                # Off-diagonals
                for band in range(1, bandwidth):
                    if(j < grid-band):
                        coo_1[i] = single_index_three(j+band, k, l, grid)
                        coo_2[i] = jkl
                        coo_data[i] = ke_bands[band]
                        i += 1

                    if(j > band-1):
                        coo_1[i] = single_index_three(j-band, k, l, grid)
                        coo_2[i] = jkl
                        coo_data[i] = ke_bands[band]
                        i += 1
 
                    if(k < grid-band):
                        coo_1[i] = single_index_three(j, k+band, l, grid)
                        coo_2[i] = jkl
                        coo_data[i] = ke_bands[band]
                        i += 1

                    if(k > band-1):
                        coo_1[i] = single_index_three(j, k-band, l, grid)
                        coo_2[i] = jkl
                        coo_data[i] = ke_bands[band]
                        i += 1

                    if(l < grid-band):
                        coo_1[i] = single_index_three(j, k, l+band, grid)
                        coo_2[i] = jkl
                        coo_data[i] = ke_bands[band]
                        i += 1

                    if(l > band-1):
                        coo_1[i] = single_index_three(j, k, l-band, grid)
                        coo_2[i] = jkl
                        coo_data[i] = ke_bands[band]
                        i += 1


    for l in range(grid-bandwidth+1, grid):
        for j in range(bandwidth-1, grid-bandwidth+1):
            for k in range(bandwidth-1, grid-bandwidth+1):

                # Main diagonal
                jkl = single_index_three(j, k, l, grid)
                coo_1[i] = jkl
                coo_2[i] = jkl
                coo_data[i] = 3.0*ke_bands[0] + v_ext[j] + v_ext[k] + v_ext[l] + v_int[j,k] + v_int[j,l] + v_int[k,l]
                i += 1

                # Off-diagonals
                for band in range(1, bandwidth):
                    if(j < grid-band):
                        coo_1[i] = single_index_three(j+band, k, l, grid)
                        coo_2[i] = jkl
                        coo_data[i] = ke_bands[band]
                        i += 1

                    if(j > band-1):
                        coo_1[i] = single_index_three(j-band, k, l, grid)
                        coo_2[i] = jkl
                        coo_data[i] = ke_bands[band]
                        i += 1
 
                    if(k < grid-band):
                        coo_1[i] = single_index_three(j, k+band, l, grid)
                        coo_2[i] = jkl
                        coo_data[i] = ke_bands[band]
                        i += 1

                    if(k > band-1):
                        coo_1[i] = single_index_three(j, k-band, l, grid)
                        coo_2[i] = jkl
                        coo_data[i] = ke_bands[band]
                        i += 1

                    if(l < grid-band):
                        coo_1[i] = single_index_three(j, k, l+band, grid)
                        coo_2[i] = jkl
                        coo_data[i] = ke_bands[band]
                        i += 1

                    if(l > band-1):
                        coo_1[i] = single_index_three(j, k, l-band, grid)
                        coo_2[i] = jkl
                        coo_data[i] = ke_bands[band]
                        i += 1


    return np.asarray(coo_1, dtype=int), np.asarray(coo_2, dtype=int), np.asarray(coo_data, dtype=np.float)


def imag_pot_two(pm):
    r"""Calculates the perturbing potential to be added to the main diagonal of 
    the two-electron Hamiltonian matrix if imaginary boundary conditions are 
    used.

    parameters
    ----------
    pm : object
        Parameters object

    returns array_like
        1D array of the perturbing potential 
    """
    # Variable declarations
    cdef int i, j, k
    cdef int grid = pm.space.npt
    cdef double complex [:] v_pert = pm.space.v_pert
    cdef double complex [:] imag_pot = np.zeros((grid**2), dtype=np.cfloat)

    # Loop over each element 
    i = 0
    for j in range(grid):
        for k in range(grid):

            # Calculate element
            imag_pot[i] = v_pert[j] + v_pert[k]
  
            # Increase count
            i += 1

    return np.asarray(imag_pot, dtype=np.cfloat)


def imag_pot_three(pm):
    r"""Calculates the perturbing potential to be added to the main diagonal of 
    the three-electron Hamiltonian matrix if imaginary boundary conditions are 
    used.

    parameters
    ----------
    pm : object
        Parameters object

    returns array_like
        1D array of the perturbing potential 
    """
    # Variable declarations
    cdef int i, j, k, l
    cdef int grid = pm.space.npt
    cdef double complex [:] v_pert = pm.space.v_pert
    cdef double complex [:] imag_pot = np.zeros((grid**3), dtype=np.cfloat)

    # Loop over each element 
    i = 0
    for j in range(grid):
        for k in range(grid):
            for l in range(grid):

                # Calculate element
                imag_pot[i] = v_pert[j] + v_pert[k] + v_pert[l]
  
                # Increase count
                i += 1

    return np.asarray(imag_pot, dtype=np.cfloat)


def change_pot_two(pm, double[:] deltav_ext):
    r"""Calculates the change in the main diagonal of the two-electron 
    Hamiltonian matrix from the change in the external potential if running the
    OPT code.

    parameters
    ----------
    pm : object
        Parameters object
    dv_ext : array_like
        1D array of the change in the external potential

    returns array_like
        1D array of the change in the main diagonal of the Hamiltonian matrix
    """
    # Variable declarations
    cdef int i, j, k
    cdef int grid = pm.space.npt
    cdef double [:] deltaH = np.zeros((grid**2), dtype=np.cfloat)

    # Loop over each element 
    i = 0
    for j in range(grid):
        for k in range(grid):

            # Calculate element
            deltaH[i] = deltav_ext[j] + deltav_ext[k]
  
            # Increase count
            i += 1

    return np.asarray(deltaH, dtype=np.cfloat)


def change_pot_three(pm, double[:] deltav_ext):
    r"""Calculates the change in the main diagonal of the three-electron 
    Hamiltonian matrix from the change in the external potential if running the
    OPT code.

    parameters
    ----------
    pm : object
        Parameters object
    dv_ext : array_like
        1D array of the change in the external potential

    returns array_like
        1D array of the change in the main diagonal of the Hamiltonian matrix
    """
    # Variable declarations
    cdef int i, j, k, l
    cdef int grid = pm.space.npt
    cdef double [:] deltaH = np.zeros((grid**3), dtype=np.cfloat)

    # Loop over each element 
    i = 0
    for j in range(grid):
        for k in range(grid): 
            for l in range(grid):

                # Calculate element
                deltaH[i] = deltav_ext[j] + deltav_ext[k] + deltav_ext[l]
  
                # Increase count
                i += 1

    return np.asarray(deltaH, dtype=np.cfloat)


def single_index_two(int j, int k, int grid):
    r"""Takes every permutation of the two electron indices and creates a 
    single unique index.

    parameters
    ----------
    j : integer
        First electron index
    k : integer
        Second electron index
    grid : integer
        Number of spatial grid points in the system

    returns integer
        Single unique index
    """
    # Variable declaration
    cdef int jk

    # Unique index
    jk = k + j*grid

    return jk


def single_index_three(int j, int k, int l, int grid):
    r"""Takes every permutation of the three electron indices and creates a 
    single unique index.

    parameters
    ----------
    j : integer
        First electron index
    k : integer
        Second electron index
    l : integer
        Third electron index
    grid : integer
        Number of spatial grid points in the system

    returns integer
        Single unique index
    """
    # Variable declaration
    cdef int jkl

    # Unique index
    jkl = l + k*grid + j*grid**2

    return jkl
    
