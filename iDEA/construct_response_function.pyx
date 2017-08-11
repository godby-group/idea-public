import numpy as np


def exact_two(double[:,:] psi_gs, double[:,:] psi_es):
    r"""Calculates the exact density response function for a two electron 
    system.

    parameters
    ----------
    psi_gs : array_like
        2D array
    psi_es : array_like
        2D array

    returns array_like
        2D array of the exact density response function, indexed as 
        response_function[space_index,space_index]
    """
    # Variable declarations
    cdef int grid = psi_gs.shape[0]
    cdef int j, k
    cdef double amplitude 
    cdef double[:] expectation_value = np.zeros((grid), dtype=np.float)
    cdef double complex [:,:] response_function = np.zeros((grid, grid), dtype=np.cfloat)

    # Calculate the expectation value for each value of x
    for j in range(grid):
        for k in range(grid):
            expectation_value[j] += 4.0*psi_gs[k,j]*psi_es[k,j] 

    # Calculate the main diagonal elements of the response function
    for j in range(grid):
        amplitude = (expectation_value[j])**2
        response_function[j,j] = amplitude/1.0j

    # Calculate the off-diagonal elements of the response function
    for j in range(grid):
        for k in range(j):
            amplitude = expectation_value[j]*expectation_value[k]
            response_function[j,k] = amplitude/1.0j 
            response_function[k,j] = amplitude/1.0j

    return response_function 


def exact_three(double[:,:,:] psi_gs, double[:,:,:] psi_es):
    r"""Calculates the exact density response function for a three electron
    system.

    parameters
    ----------


    returns array_like

    """
    # Variable declarations
    cdef int grid = psi_gs.shape[0]
    cdef double complex [:,:] response_function = np.zeros((grid, grid), dtype=np.cfloat)

    return response_function


#def ks():
#    r"""Calculates the non-interacting density response function.
#
#    parameters
#    ----------
#
#    returns array_like
#        2D array of the non-interacting  density response function, indexed as 
#        response_function[space_index,space_index]
#    """
#    # Variable declarations
#    cdef int grid = psi_occ.shape[0]
#    cdef int j, k
#    cdef double amplitude
#    cdef double complex [:,:] response_function = np.zeros((grid, grid), dtype=np.cfloat)
#
#    # Calculate the amplitudes and then the response function
#    for j in range(grid):
#        for k in range(j):
#            amplitude = psi_occ[j]*psi_unocc[j]*psi_occ[k]*psi_unocc[k]
#            response_function[j,k] = amplitude/1.0j
#            response_function[k,j] = amplitude/1.0j
#
#    return response_function
