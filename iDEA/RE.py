"""Calculates the exact Kohn-Sham potential and exchange-correlation potential
for a given electron density using the reverse-engineering algorithm. This
works for both a ground-state density and a time-dependent density.
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import math
import pickle
import numpy as np
import scipy.sparse as sps
import scipy.linalg as spla
import scipy.sparse.linalg as spsla
from scipy.optimize import curve_fit
from . import RE_Utilities
from . import results as rs


def read_input_density(pm, approx):
    r"""Reads in the electron density that was calculated using the selected
    approximation.

    parameters
    ----------
    pm : object
        Parameters object
    approx : string
        The approximation used to calculate the electron density

    returns array_like
        2D array of the ground-state/time-dependent electron density from the
        approximation, indexed as density_approx[time_index,space_index]
    """
    if(pm.run.time_dependence == True):
        density_approx = np.zeros((pm.sys.imax,pm.sys.grid), dtype=np.float)
        name = 'td_{}_den'.format(approx)
        density_approx[:,:] = rs.Results.read(name, pm)
    else:
        density_approx = np.zeros((1,pm.sys.grid), dtype=np.float)
        name = 'gs_{}_den'.format(approx)
        density_approx[0,:] = rs.Results.read(name, pm)

    return density_approx


def read_input_current_density(pm, approx):
    r"""Reads in the electron current density that was calculated using the
    selected approximation.

    parameters
    ----------
    pm : object
        Parameters object
    approx : string
        The approximation used to calculate the electron current density

    returns array_like
        2D array of the electron current density from the approximation,
        indexed as current_density_approx[time_index,space_index]
    """
    current_density_approx = np.zeros((pm.sys.imax,pm.sys.grid),
                             dtype=np.float)
    name = 'td_{}_cur'.format(approx)
    current_density_approx[:,:] = rs.Results.read(name, pm)

    return current_density_approx


def construct_kinetic_energy(pm):
    r"""Stores the band elements of the kinetic energy matrix in lower form.
    The kinetic energy matrix is constructed using a three-point, five-point
    or seven-point stencil. This yields an NxN band matrix (where N is the
    number of grid points). For example with N=6 and a three-point stencil:

    .. math::

        K = -i \frac{d^2}{dx^2}=
        -\frac{1}{2} \begin{pmatrix}
        -2 & 1 & 0 & 0 & 0 & 0 \\
        1 & -2 & 1 & 0 & 0 & 0 \\
        0 & 1 & -2 & 1 & 0 & 0 \\
        0 & 0 & 1 & -2 & 1 & 0 \\
        0 & 0 & 0 & 1 & -2 & 1 \\
        0 & 0 & 0 & 0 & 1 & -2
        \end{pmatrix}
        \frac{1}{\delta x^2}
        = \frac{1}{\delta x^2} \bigg[1,-\frac{1}{2}\bigg]

    parameters
    ----------
    pm : object
        Parameters object

    returns array_like
        2D array of the kinetic energy matrix, index as
        kinetic_energy[band,space_index]
    """
    sd = pm.space.second_derivative_band
    nbnd = len(sd)
    kinetic_energy = np.zeros((nbnd, pm.sys.grid), dtype=np.float)

    for i in range(nbnd):
        kinetic_energy[i,:] = -0.5 * sd[i]

    return kinetic_energy


def construct_momentum(pm):
    r"""Stores the lower band elements of the momentum matrix.
    The momentum matrix is constructed using a five-point
    or seven-point stencil. This yields an NxN band matrix (where N is the
    number of grid points). For example with N=6 and a five-point stencil:

    .. math::

        p = -i \frac{d}{dx}=
        -\frac{i}{12} \begin{pmatrix}
        0 & 8 & -1 & 0 & 0 & 0 \\
        -8 & 0 & 8 & -1 & 0 & 0 \\
        1 & -8 & 0 & 8 & -1 & 0 \\
        0 & 1 & -8 & 0 & 8 & -1 \\
        0 & 0 & 1 & -8 & 0 & 8 \\
        0 & 0 & 0 & 1 & -8 & 0
        \end{pmatrix}
        \frac{1}{\delta x}
        = \frac{1}{\delta x} \bigg[0,\frac{2}{3},-\frac{1}{12}\bigg]

    parameters
    ----------
    pm : object
        Parameters object

    returns array_like
        1D array of the lower band elements of the momentum matrix, index as
        momentum[band]
    """
    fd = pm.space.first_derivative_band
    nbnd = len(fd)
    momentum = np.zeros(nbnd, dtype=np.cfloat)

    for i in range(nbnd):
        momentum[i] = -1.0j * fd[i]

    return momentum


def construct_damping(pm):
    r"""Stores the damping function which is used to filter out noise in the
    Kohn-Sham vector potential.

    .. math::

        f_{\mathrm{damping}}(x) = e^{-\alpha x^{2}}

    parameters
    ----------
    pm : object
        Parameters object

    returns array_like
        1D array of the damping function used to filter out noise, indexed as
        damping[frequency_index]
    """
    # Only valid for x>=0
    damping_length = int((pm.sys.grid+1)/2)
    damping = np.zeros(damping_length, dtype=np.float)

    # Exponential damping term
    for j in range(damping_length):
        x = j*pm.sys.deltax
        damping[j] = np.exp(-pm.re.damping*(x**2.0))

    return damping


def construct_A_initial(pm, kinetic_energy, v_ks):
    r"""Constructs the sparse matrix A at t=0, once the external perturbation
    has been applied.

    .. math::

        A_{\mathrm{initial}} = I + i\dfrac{\delta t}{2}H(t=0) \\
        H(t=0) = K + V_{\mathrm{KS}}(t=0)

    parameters
    ----------
    pm : object
        Parameters object
    kinetic_energy : array_like
        2D array of the kinetic energy matrix, index as
        kinetic_energy[band,space_index]
    v_ks : array_like
        1D array of the ground-state Kohn-Sham potential + the external
        perturbation, indexed as v_ks[space_index]

    returns sparse_matrix
        The sparse matrix A at t=0, A_initial, used when solving the equation
        Ax=b
    """
    A_initial = np.zeros((pm.sys.grid, pm.sys.grid), dtype=np.cfloat)
    nbnd = kinetic_energy.shape[0]
    prefactor = 0.5j*pm.sys.deltat

    # Assign the main-diagonal elements
    for j in range(pm.sys.grid):
        A_initial[j,j] = prefactor*(kinetic_energy[0,j] + v_ks[j])

        # Assign the off-diagonal elements
        for k in range(1, nbnd):
            if((j+k) < pm.sys.grid):
                A_initial[j,j+k] = prefactor*kinetic_energy[k,j]
                A_initial[j+k,j] = prefactor*kinetic_energy[k,j]

    # Add the identity matrix
    A_initial += sps.identity(pm.sys.grid, dtype=np.cfloat)

    return A_initial


def construct_A(pm, A_initial, A_ks, momentum):
    r"""Constructs the sparse matrix A at time t.

    .. math::

        A = I + i\dfrac{\delta t}{2}H \\
        H = \frac{1}{2}(p-A_{\mathrm{KS}})^{2} + V_{\mathrm{KS}}(t=0) \\
        = K + V_{\mathrm{KS}}(t=0) + \frac{A_{\mathrm{KS}}^{2}}{2} -
        \frac{pA_{\mathrm{KS}}}{2} - \frac{A_{\mathrm{KS}}p}{2} \\
        = H(t=0) + \frac{A_{\mathrm{KS}}^{2}}{2} - \frac{pA_{\mathrm{KS}}}{2}
        - \frac{A_{\mathrm{KS}}p}{2}

    parameters
    ----------
    pm : object
        Parameters object
    A_initial : sparse_matrix
        The sparse matrix A at t=0
    A_ks : array_like
        1D array of the time-dependent Kohn-Sham vector potential, at time t,
        indexed as A_ks[space_index]
    momentum : array_like
        1D array of the lower band elements of the momentum matrix, index as
        momentum[band,space_index]

    returns sparse_matrix
        The sparse matrix A at time t
    """
    A = np.copy(A_initial)
    nbnd = momentum.shape[0]
    prefactor = 0.25j*pm.sys.deltat

    # Assign the main diagonal elements
    for j in range(pm.sys.grid):
        A[j,j] += prefactor*(A_ks[j]**2)

        # Assign the off-diagonal elements
        for k in range(1, nbnd):
            if((j+k) < pm.sys.grid):
                A[j+k,j] -= prefactor*(momentum[k]*A_ks[j])
                A[j,j+k] += prefactor*(A_ks[j]*momentum[k])
            if((j-k) >= 0):
                A[j-k,j] += prefactor*(momentum[k]*A_ks[j])
                A[j,j-k] -= prefactor*(A_ks[j]*momentum[k])

    # Convert to compressed sparse column (csc) format for efficient arithmetic
    A = sps.csc_matrix(A)

    return A


def calculate_ground_state(pm, approx, density_approx, v_ext, kinetic_energy):
    r"""Calculates the exact ground-state Kohn-Sham potential by solving the
    ground-state Kohn-Sham equations and iteratively correcting v_ks. The
    exact ground-state Kohn-Sham eigenfunctions, eigenenergies and electron
    density are then calculated.

    .. math::

        V_{\mathrm{KS}}(x) \rightarrow V_{\mathrm{KS}}(x) +
        \mu[n_{\mathrm{KS}}^{p}(x)-n_{\mathrm{approx}}^{p}(x)]

    parameters
    ----------
    pm : object
        Parameters object
    approx : string
        The approximation used to calculate the electron density
    density_approx : array_like
        1D array of the ground-state electron density from the approximation,
        indexed as density_approx[space_index]
    v_ext : array_like
        1D array of the unperturbed external potential, indexed as
        v_ext[space_index]
    kinetic_energy : array_like
        2D array of the kinetic energy matrix, index as
        kinetic_energy[band,space_index]

    returns array_like and array_like and array_like and array_like and Boolean
        1D array of the ground-state Kohn-Sham potential, indexed as
        v_ks[space_index]. 1D array of the ground-state Kohn-Sham electron
        density, indexed as density_ks[space_index]. 2D array of the
        ground-state Kohn-Sham eigenfunctions, indexed as
        wavefunctions_ks[space_index,eigenfunction]. 1D array containing the
        ground-state Kohn-Sham eigenenergies, indexed as
        energies_ks[eigenenergies]. Boolean - True if file containing exact
        Kohn-Sham potential is found, False if file is not found.
    """
    # Read in v_ks or take the external potential as the initial guess
    string = 'RE: calculating the ground-state Kohn-Sham potential for the' + \
             ' {} density'.format(approx)
    pm.sprint(string, 1, newline=True)
    v_ks = np.zeros(pm.sys.grid, dtype=np.float)
    try:
        name = 'gs_{}re_vks'.format(approx)
        v_ks[:] = rs.Results.read(name, pm)
        string = '    (found exact ground-state Kohn-Sham potential to start' + \
                 ' from)'
        pm.sprint(string, 1, newline=True)
        file_exist = True
    except:
        string = '    (starting from the external potential)'
        pm.sprint(string, 1, newline=True)
        v_ks[:] += v_ext[:]
        file_exist = False

    # Construct the Hamiltonian matrix for the initial v_ks and solve the
    # ground-state Kohn-Sham equations
    hamiltonian = np.copy(kinetic_energy)
    hamiltonian[0,:] += v_ks[:]
    wavefunctions_ks, energies_ks, density_ks = (solve_gsks_equations(
            pm, hamiltonian))
    density_difference = abs(density_approx-density_ks)
    density_error = np.trapz(density_difference, dx=pm.sys.deltax)
    string = 'RE: initial guess density error = {}'.format(density_error)
    pm.sprint(string, 1, newline=True)

    # Solve the ground-state Kohn-Sham equations and iteratively correct v_ks
    iterations = 0
    while(pm.re.mu >1e-15):

        # Save the last iteration
        density_error_old = density_error

        # Iteratively correct v_ks within the Hamiltonian matrix
        hamiltonian[0,:] += pm.re.mu*(density_ks[:]**pm.re.p -
                            density_approx[:]**pm.re.p)

        # Solve the ground-state Kohn-Sham equations
        wavefunctions_ks, energies_ks, density_ks = solve_gsks_equations(pm,
                                                    hamiltonian)

        # Calculate the error in the ground-state Kohn-Sham density
        density_difference = abs(density_approx-density_ks)
        density_error = np.trapz(density_difference, dx=pm.sys.deltax)
        string = 'RE: density error = {}'.format(density_error)
        pm.sprint(string, 1, newline=False)

        # Ensure stable convergence
        error_change = density_error - density_error_old
        if((error_change > 0.0) or (abs(error_change)<1e-15)):
            pm.re.mu /= 2.0

        iterations +=1

    # Print
    if(iterations > 0):
        pm.sprint('', 1, newline=True)
    string = 'RE: calculated the ground-state Kohn-Sham potential,'
    pm.sprint(string, 1, newline=True)
    string = '    error = {}'.format(density_error)
    pm.sprint(string, 1, newline=True)

    # Extract the exact v_ks from the Hamiltonian matrix
    v_ks[:] = hamiltonian[0,:] - kinetic_energy[0,:]

    return v_ks, density_ks, wavefunctions_ks, energies_ks, file_exist


def solve_gsks_equations(pm, hamiltonian):
    r"""Solves the ground-state Kohn-Sham equations to find the ground-state
    Kohn-Sham eigenfunctions, energies and electron density.

    .. math::

        \hat{H}\phi_{j}(x) = \varepsilon_{j}\phi_{j}(x) \\
        n_{\mathrm{KS}}(x) = \sum_{j=1}^{N}|\phi_{j}(x)|^{2}

    parameters
    ----------
    pm : object
        Parameters object
    hamiltonian : array_like
        2D array of the Hamiltonian matrix, index as
        kinetic_energy[band,space_index]

    returns array_like and array_like and array_like
        2D array of the ground-state Kohn-Sham eigenfunctions, indexed as
        wavefunctions_ks[space_index,eigenfunction]. 1D array containing the
        ground-state Kohn-Sham eigenenergies, indexed as
        energies_ks[eigenenergies]. 1D array of the ground-state Kohn-Sham
        electron density, indexed as density_ks[space_index].
    """
    # Solve the Kohn-Sham equations
    energies_ks, wavefunctions_ks = spla.eig_banded(hamiltonian, lower=True,
                                    select='i')

    # Normalise the wavefunctions
    for j in range(pm.sys.NE):
        normalisation = np.linalg.norm(wavefunctions_ks[:,j])*pm.sys.deltax**0.5
        wavefunctions_ks[:,j] /= normalisation

    # Calculate the electron density
    density_ks = np.sum(wavefunctions_ks[:,:pm.sys.NE]**2, axis=1, dtype=np.float)

    return wavefunctions_ks, energies_ks, density_ks


def calculate_time_dependence(pm, A_initial, momentum, A_ks, damping,
                              wavefunctions_ks, density_ks, density_approx,
                              current_density_approx):
    r"""Calculates the exact time-dependent Kohn-Sham vector potential, at time
    t+dt, by solving the time-dependent Kohn-Sham equations and iteratively
    correcting A_ks. The exact time-dependent Kohn-Sham eigenfunctions,
    electron density and electron current density are then calculated.

    .. math::

        A_{\mathrm{KS}}(x,t) \rightarrow A_{\mathrm{KS}}(x,t) +
        \nu\bigg[\frac{j_{\mathrm{KS}}(x,t)-j_{\mathrm{approx}}(x,t)}
        {n_{\mathrm{approx}}(x,t)}\bigg]

    parameters
    ----------
    pm : object
        Parameters object
    A_initial : sparse_matrix
        The sparse matrix A at t=0, A_initial, used when solving the equation
        Ax=b
    momentum : array_like
        1D array of the lower band elements of the momentum matrix, index as
        momentum[band]
    A_ks : array_like
        1D array of the time-dependent Kohn-Sham vector potential, at time t+dt,
        indexed as A_ks[space_index]
    damping : array_like
        1D array of the damping function used to filter out noise, indexed as
        damping[frequency_index]
    wavefunctions_ks : array_like
        2D array of the time-dependent Kohn-Sham eigenfunctions, at time t,
        indexed as wavefunctions_ks[space_index,eigenfunction]
    density_ks : array_like
        2D array of the time-dependent Kohn-Sham electron density, at time t
        and t+dt, indexed as density_ks[time_index, space_index]
    density_approx : array_like
        1D array of the time-dependent electron density from the approximation,
        at time t+dt, indexed as density_approx[space_index]
    current_density_approx : array_like
        1D array of the electron current density from the approximation, at
        time t+dt, indexed as current_density_approx[space_index]

    returns array_like and array_like and array_like and array_like and float
    and float
        1D array of the time-dependent Kohn-Sham vector potential, at time t+dt,
        indexed as A_ks[space_index]. 1D array of the time-dependent Kohn-Sham
        electron density, at time t+dt, indexed as density_ks[space_index]. 1D
        array of the Kohn-Sham electron current density, at time t+dt, indexed
        as current_density_ks[space_index]. 2D array of the time-dependent
        Kohn-Sham eigenfunctions, at time t+dt, indexed as
        wavefunctions_ks[space_index,eigenfunction]. The error between the
        Kohn-Sham electron density and electron density from the approximation.
        The error between the Kohn-Sham electron current density and electron
        current density from the approximation.
    """
    # Create an array to store the A_ks that minimises the error in the current
    # density
    A_ks_best = np.copy(A_ks)

    # Create a copy of the time-dependent Kohn-Sham eigenfunctions at the
    # beginning of the time step
    wavefunctions_ks_copy = np.copy(wavefunctions_ks)

    # Solve the time-dependent Kohn-Sham equations and iteratively correct A_ks
    iterations = 0
    while((iterations < pm.re.max_iterations)):

        # Construct the sparse matrix A
        A = construct_A(pm, A_initial, A_ks, momentum)

        # Solve the time-dependent Kohn-Sham equations
        density_ks[1,:], wavefunctions_ks = solve_tdks_equations(pm, A,
                                            wavefunctions_ks)

        # Calculate the Kohn-Sham current density
        current_density_ks = calculate_current_density(pm, density_ks)

        # Calculate the error in the Kohn-Sham charge density
        density_error = np.sum(abs(density_approx[:]-density_ks[1,:])
                        )*pm.sys.deltax

        # Calculate the error in the Kohn-Sham current density
        current_density_difference = abs(current_density_approx-current_density_ks)
        current_density_error = np.trapz(current_density_difference,
                                dx=pm.sys.deltax)
        if(iterations == 0):
            current_density_error_min = np.copy(current_density_error)

        # Store A_ks if it is an improvement
        if(current_density_error < current_density_error_min):
            A_ks_best[:] = A_ks[:]
            current_density_error_min = np.copy(current_density_error)

        # Iteratively correct A_ks
        A_ks[:] += pm.re.nu*((current_density_ks[:] - current_density_approx[:]
                   )/density_approx[:])

        # Filter out noise in A_ks
        if(pm.re.damping != 0):
            A_ks = filter_noise(pm, A_ks, damping)

        # Reset the Kohn-Sham eigenfunctions
        wavefunctions_ks[:,:] = wavefunctions_ks_copy[:,:]

        iterations += 1

    # Use the best A_ks
    A_ks[:] = A_ks_best[:]

    # Construct the sparse matrix A
    A = construct_A(pm, A_initial, A_ks, momentum)

    # Solve the time-dependent Kohn-Sham equations
    density_ks[1,:], wavefunctions_ks = solve_tdks_equations(pm, A,
                                        wavefunctions_ks)

    # Calculate the Kohn-Sham current density
    current_density_ks = calculate_current_density(pm, density_ks)

    # Calculate the error in the Kohn-Sham charge density
    density_error = np.sum(abs(density_approx[:]-density_ks[1,:])
                    )*pm.sys.deltax

    # Calculate the error in the Kohn-Sham current density
    current_density_difference = abs(current_density_approx-current_density_ks)
    current_density_error = np.trapz(current_density_difference,
                            dx=pm.sys.deltax)

    return (A_ks, density_ks[1,:], current_density_ks, wavefunctions_ks,
           density_error, current_density_error)


def solve_tdks_equations(pm, A, wavefunctions_ks):
    r"""Solves the time-dependent Kohn-Sham equations to find the
    time-dependent Kohn-Sham eigenfunctions and electron density.

    .. math::

        \hat{H}\phi_{j}(x,t) = i\frac{\partial\phi_{j}(x,t)}{\partial t} \\
        n_{\mathrm{KS}}(x,t) = \sum_{j=1}^{N}|\phi_{j}(x,t)|^{2}

    parameters
    ----------
    pm : object
        Parameters object
    A : sparse_matrix
        The sparse matrix A, used when solving the equation Ax=b
    wavefunctions_ks : array_like
        2D array of the time-dependent Kohn-Sham eigenfunctions, at time t+dt,
        indexed as wavefunctions_ks[space_index,eigenfunction]

    returns array_like and array_like
        1D array of the time-dependent Kohn-Sham electron density, at time
        t+dt, indexed as density_ks[space_index]. 2D array of the
        time-dependent Kohn-Sham eigenfunctions, at time t+dt, indexed as
        wavefunctions_ks[space_index,eigenfunction]
    """
    # Construct the sparse matrix C
    C = 2.0*sps.identity(pm.sys.grid, dtype=np.cfloat) - A

    # Loop over each electron
    for j in range(pm.sys.NE):

        # Construct the vector b for each wavefunction
        b = C*wavefunctions_ks[:,j]

        # Solve Ax=b
        wavefunctions_ks[:,j], info = spsla.cg(A,b,x0=wavefunctions_ks[:,j],
                                      tol=pm.re.rtol_solver)

    # Calculate the electron density
    density_ks = np.sum(abs(wavefunctions_ks[:,:pm.sys.NE])**2, axis=1, dtype=np.float)

    return density_ks, wavefunctions_ks


def filter_noise(pm, A_ks, damping):
    r"""Filters out noise in the Kohn-Sham vector potential by suppressing
    high-frequency terms in the Fourier transform.

    parameters
    ----------
    pm : object
        Parameters object
    A_ks : array_like
        1D array of the time-dependent Kohn-Sham vector potential, at time t+dt,
        indexed as A_ks[space_index]
    damping : array_like
        1D array of the damping function used to filter out noise, indexed as
        damping[frequency_index]

    returns array_like
        1D array of the time-dependent Kohn-Sham vector potential, at time t+dt,
        indexed as A_ks[space_index]
    """
    # Calculate the Fourier transform of A_ks
    A_ks_freq = np.fft.rfft(A_ks)

    # Apply the damping function to suppress high-frequency terms
    A_ks_freq[:] *= damping[:]

    # Calculate the inverse Fourier transform to recover the spatial
    # representation of A_ks with the noise filtered out
    A_ks[:] = np.fft.irfft(A_ks_freq, len(A_ks))

    return A_ks


def remove_gauge(pm, A_ks, v_ks, v_ks_gs):
    r"""Removes the gauge transformation that was applied to the Kohn-Sham
    potential, so that it becomes a fully scalar quantity.

    .. math::

        V_{\mathrm{KS}}(x,t) \rightarrow V_{\mathrm{KS}}(x,t) +
        \int_{-x_{\mathrm{max}}}^{x}
        \frac{\partial A_{\mathrm{KS}}(x',t)}{\partial t} dx'

    parameters
    ----------
    pm : object
        Parameters object
    A_ks : array_like
        2D array of the time-dependent Kohn-Sham vector potential, at time t
        and t+dt, indexed as A_ks[time_index, space_index]
    v_ks : array_like
        1D array of the time-dependent Kohn-Sham potential, at time t+dt,
        indexed as v_ks[space_index]
    v_ks_gs : array_like
        1D array of the ground-state Kohn-Sham potential, indexed as
        v_ks[space_index]

    returns array_like
        1D array of the time-dependent Kohn-Sham potential, at time t +dt,
        indexed as v_ks[space_index]
    """
    # Change gauge to calculate the full Kohn-Sham (scalar) potential
    for j in range(pm.sys.grid):
        for k in range(j):
            v_ks[:] += (A_ks[1,k] - A_ks[0,k])*(pm.sys.deltax/pm.sys.deltat)

    # Shift the Kohn-Sham potential to match the ground-state Kohn-Sham
    # potential at the centre of the system
    shift = v_ks_gs[int((pm.sys.grid-1)/2)] - v_ks[int((pm.sys.grid-1)/2)]
    v_ks[:] += shift

    return v_ks[:]


def calculate_hartree_potential(pm, density):
    r"""Calculates the Hartree potential for a given electron density.

    .. math::

        V_{\mathrm{H}}(x) = \int U(x,x') n(x')dx'

    parameters
    ----------
    pm : object
        Parameters object
    density_ks : array_like
        1D array of the Kohn-Sham electron density, either the ground-state or
        at a particular time step, indexed as density_ks[space_index]

    returns array_like
        1D array of the Hartree potential for a given electron density, indexed
        as v_h[space_index]
    """

    return np.dot(pm.space.v_int,density)*pm.sys.deltax


def calculate_current_density(pm, density_ks):
    r"""Calculates the Kohn-Sham electron current density, at time t+dt, from
    the time-dependent Kohn-Sham electron density by solving the continuity
    equation.

    .. math::

        \frac{\partial n_{\mathrm{KS}}}{\partial t} + \nabla
        \cdot j_{\mathrm{KS}} = 0

    parameters
    ----------
    pm : object
        Parameters object
    density_ks : array_like
        2D array of the time-dependent Kohn-Sham electron density, at time t
        and t+dt, indexed as density_ks[time_index, space_index]

    returns array_like
        1D array of the Kohn-Sham electron current density, at time t+dt,
        indexed as current_density_ks[space_index]
    """
    current_density_ks = np.zeros(pm.sys.grid, dtype=np.float, order='F')
    current_density_ks = RE_Utilities.continuity_eqn(current_density_ks,
                         density_ks[1,:], density_ks[0,:], pm.sys.deltax,
                         pm.sys.deltat, pm.sys.grid)

    return current_density_ks


def xc_correction(pm, v_xc, x_points):
    r"""Calculates an approximation to the constant that needs to be added to
    the exchange-correlation potential so that it asymptotically approaches
    zero at large |x|. The approximate error (standard deviation) on the
    constant is also calculated.

    .. math::

        V_{\mathrm{xc}}(x) \rightarrow V_{\mathrm{xc}}(x) + a \ , \ \
        \text{s.t.} \ \lim_{|x| \to \infty} V_{\mathrm{xc}}(x) = 0

    parameters
    ----------
    pm : object
        Parameters object
    v_xc : array_like
        1D array of the ground-state exchange-correlation potential, indexed as
        v_xc[space_index]
    x_points : array_like
        1D array of the spatial grid

    returns float and float
        An approximation to the constant that needs to be added to the
        exchange-correlation potential. The approximate error (standard
        deviation) on the constant.
    """
    # The range over which the fit will be applied
    x_min = int(0.05*pm.sys.grid)
    x_max = int(0.15*pm.sys.grid)

    # Calculate the fit and determine the correction to v_xc and its error
    fit, variance = curve_fit(xc_fit, x_points[x_min:x_max], v_xc[x_min:x_max])
    correction = fit[0]
    correction_error = variance[0,0]**0.5

    string = 'RE: correction to the asymptotic form of V_xc = {},'\
             .format(correction)
    pm.sprint(string, 1, newline=True)
    string = '    error = {}'.format(correction_error)
    pm.sprint(string, 1, newline=True)

    return correction, correction_error


def xc_fit(x_points, correction):
    r"""Applies a fit to the exchange-correlation potential over a specified
    range near the edge of the system's grid to determine the correction that
    needs to be applied to give the correct asymptotic behaviour at large |x|

    .. math::

        V_{\mathrm{xc}}(x) \approx \frac{1}{x} + a

    parameters
    ----------
    x_points : array_like
        1D array of the spatial grid over a specified range near the edge of
        the system
    correction : float
        An approximation to the constant that needs to be added to the
        exchange-correlation potential

    returns array_like
        A fit to the exchange-correlation potential over the specified range
    """

    return 1.0/x_points + correction


def calculate_xc_energy(pm, approx, density_ks, v_h, v_xc, energies_ks):
    r"""Calculates the exchange-correlation energy of the ground-state system.

    .. math::

        E_{\mathrm{xc}} = E_{\mathrm{total}} - \sum_{j=1}^{N}\varepsilon_{j} +
        \int \bigg[\frac{1}{2}V_{\mathrm{H}}(x) +
        V_{\mathrm{xc}}(x)\bigg]n_{\mathrm{KS}}(x)dx

    parameters
    ----------
    pm : object
        Parameters object
    approx : string
        The approximation used to calculate the electron density
    density_ks : array_like
        1D array of the ground-state Kohn-Sham electron density, indexed as
        density_ks[space_index]
    v_h : array_like
        1D array of the ground-state Hartree potential, indexed as
        v_h[space_index]
    v_xc : array_like
        1D array of the ground-state exchange-correlation potential, indexed as
        v_xc[space_index]
    energies_ks : array_like
        1D array containing the ground-state Kohn-Sham eigenenergies, indexed
        as energies_ks[eigenenergies]

    returns float
        The exchange-correlation energy of the ground-state system
    """
    try:
        name = 'gs_{}_E'.format(approx)
        energy_approx = rs.Results.read(name, pm)
        E_xc = energy_approx - np.sum(energies_ks)
        for j in range(pm.sys.grid):
            E_xc += (density_ks[j])*(0.5*v_h[j] + v_xc[j])*pm.sys.deltax
    except:
        E_xc = 0.0
        string = 'RE: the exchange-correlation energy could not be ' + \
                 'calculated as no file containing the total energy of ' + \
                 'the system could be found'
        pm.sprint(string, 1, newline=True)

    return E_xc


def main(parameters, approx):
    r"""Calculates the exact Kohn-Sham potential and exchange-correlation
    potential for a given electron density using the reverse-engineering
    algorithm. This works for both a ground-state density and a time-dependent
    density.ground-state of the system.

    parameters
    ----------
    parameters : object
        Parameters object
    approx : string
        The approximation used to calculate the electron density

    returns object
        Results object
    """
    pm = parameters
    pm.setup_space()

    # Array initialisations
    string = 'RE: constructing arrays'
    pm.sprint(string, 1, newline=True)
    x_points = pm.space.grid
    v_ext = pm.space.v_ext
    kinetic_energy = construct_kinetic_energy(pm)
    if(pm.run.time_dependence == True):
        density_ks = np.zeros((pm.sys.imax,pm.sys.grid), dtype=np.float)
    else:
        density_ks = np.zeros((1,pm.sys.grid), dtype=np.float)
    v_ks = np.copy(density_ks)
    v_hxc = np.copy(density_ks)
    v_h = np.copy(density_ks)
    v_xc = np.copy(density_ks)

    # Read the density calculated using the approximation
    density_approx = read_input_density(pm, approx)

    # Calculate the ground-state Kohn-Sham potential, density, occupied
    # eigenfunctions and their eigenenergies
    v_ks[0,:], density_ks[0,:], wavefunctions_ks, energies_ks, file_exist = (
            calculate_ground_state(pm, approx, density_approx[0,:], v_ext,
            kinetic_energy))

    # Calculate the ground-state Hartree potential, exchange-correlation
    # potential and Hartree-exchange-correlation potential
    v_hxc[0,:] = v_ks[0,:] - v_ext[:]
    v_h[0,:] = calculate_hartree_potential(pm, density_ks[0,:])
    v_xc[0,:] = v_hxc[0,:] - v_h[0,:]

    # Correct the asymptotic form of v_xc and v_ks
    if(file_exist == False):
        correction, correction_error = xc_correction(pm, v_xc[0,:], x_points)
        v_ks[0,:] -= correction
        v_xc[0,:] -= correction
        energies_ks[:] -= correction

    # Calculate the exchange-correlation energy
    E_xc = calculate_xc_energy(pm, approx, density_ks[0,:], v_h[0,:],
                               v_xc[0,:], energies_ks)

    # Save the ground-state quantities to file
    approxre = approx + 're'
    results = rs.Results()
    results.add(density_ks[0,:],'gs_{}_den'.format(approxre))
    results.add(v_ks[0,:],'gs_{}_vks'.format(approxre))
    results.add(v_hxc[0,:],'gs_{}_vhxc'.format(approxre))
    results.add(v_h[0,:],'gs_{}_vh'.format(approxre))
    results.add(v_xc[0,:],'gs_{}_vxc'.format(approxre))
    results.add(E_xc,'gs_{}_Exc'.format(approxre))
    if(pm.re.save_eig):
        results.add(wavefunctions_ks.T,'gs_{}_eigf'.format(approxre))
        results.add(energies_ks,'gs_{}_eigv'.format(approxre))
    if(pm.run.save):
        results.save(pm)

    # Reverse-engineer the time-dependent system
    if(pm.run.time_dependence == True):

        # Array initialisations
        string = 'RE: constructing arrays'
        pm.sprint(string, 1, newline=True)
        wavefunctions_ks = wavefunctions_ks.astype(np.cfloat)
        if(pm.sys.im == 1):
            v_ks = v_ks.astype(np.cfloat)
            v_ext = v_ext.astype(np.cfloat)
            v_xc = v_xc.astype(np.cfloat)
        v_ext += pm.space.v_pert
        v_ks[1:,:] += (v_ks[0,:] + pm.space.v_pert)
        A_ks = np.zeros((2,pm.sys.grid), dtype=np.float)
        current_density_ks = np.zeros((pm.sys.imax,pm.sys.grid), dtype=np.float)
        momentum = construct_momentum(pm)
        damping = construct_damping(pm)
        A_initial = construct_A_initial(pm, kinetic_energy, v_ks[1,:])

        # Read the current density calculated using the approximation
        current_density_approx = read_input_current_density(pm, approx)

        # Reverse-engineer each time step
        for i in range(1, pm.sys.imax):

            # Use A_ks from the last time step as the initial guess
            A_ks[1,:] = A_ks[0,:]

            # Calculate the Kohn-Sham vector potential, the density, the
            # current density, the eigenfunctions and the errors between the
            # Kohn-Sham and approximation densities and current densities
            A_ks[1,:], density_ks[i,:], current_density_ks[i,:], wavefunctions_ks, density_error, current_density_error = (
                    calculate_time_dependence(pm, A_initial, momentum, A_ks[1,:],
                    damping, wavefunctions_ks, density_ks[i-1:i+1,:],
                    density_approx[i,:], current_density_approx[i,:]))

            # Print to screen
            string = 'RE: t = {0}, current density error = {1}, density error = {2}'\
                     .format(i*pm.sys.deltat, current_density_error, density_error)
            pm.sprint(string,1,newline=False)

            # If the required tolerance has been reached
            if((density_error < pm.re.density_tolerance) and
            (current_density_error < pm.re.cdensity_tolerance)):

                # Remove the gauge to get the full Kohn-Sham scalar potential
                v_ks[i,:] = remove_gauge(pm, A_ks, v_ks[i,:], v_ks[0,:])

                # Calculate the Hartree-potential, exchange-correlation
                # potential and Hartree-exchange-correlation potential
                v_hxc[i,:] = v_ks[i,:] - v_ext[:]
                v_h[i,:] = calculate_hartree_potential(pm, density_ks[i,:])
                v_xc[i,:] = v_hxc[i,:] - v_h[i,:]

            else:
                pm.sprint('', 1, newline=True)
                string = 'RE: The minimum tolerance has not been met.' + \
                ' Stopping at t = {}'.format(i*pm.sys.deltat)
                pm.sprint(string, 1, newline=True)
                break

            # Save the current time step
            A_ks[0,:] = A_ks[1,:]

            # Print to screen
            if(i == pm.sys.imax-1):
                pm.sprint('', 1, newline=True)

        # Save the time-dependent quantities to file
        results.add(density_ks,'td_{}_den'.format(approxre))
        results.add(current_density_ks,'td_{}_cur'.format(approxre))
        results.add(v_ks,'td_{}_vks'.format(approxre))
        results.add(v_hxc,'td_{}_vhxc'.format(approxre))
        results.add(v_h,'td_{}_vh'.format(approxre))
        results.add(v_xc,'td_{}_vxc'.format(approxre))
        if(pm.run.save):
            results.save(pm)

    return results
