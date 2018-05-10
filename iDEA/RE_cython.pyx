"""Contains the cython modules that are called within RE. Cython is used for
operations that are very expensive to do in Python, and performance speeds are
close to C.
"""
cimport numpy as np
import numpy as np

#cython: boundscheck=False, wraparound=False, nonecheck=False


def continuity_eqn(object pm, np.ndarray density_new, np.ndarray density_old):
    r"""Calculates the electron current density of the system for a particular
    time step by solving the continuity equation.

    parameters
    ----------
    pm : object
        Parameters object
    density_new : array_like
        1D array of the electron density at time t
    density_old : array_like
        1D array of the electron density at time t-dt

    returns array_like
        1D array of the electron current density at time t
    """
    # Variable declarations
    cdef int j, k
    cdef int grid = pm.space.npt
    cdef double deltax = pm.space.delta
    cdef double deltat = pm.sys.deltat
    cdef double prefactor 
    cdef np.ndarray current_density = np.zeros(pm.sys.grid, dtype=np.float)

    # Parameters
    prefactor = deltax/deltat

    # Solve the continuity equation
    for j in range(1, grid):
        for k in range(j):
            current_density[j] = current_density[j] - prefactor*(density_new[k]-density_old[k])

    return current_density
