"""Calculates the exact external potential and energy for a system of one, two 
or three interacting electrons with a specified electron density.
"""
from __future__ import division
from __future__ import absolute_import

import numpy as np
import scipy as sp
import scipy.sparse as sps
import scipy.linalg as spla
from . import results as rs
from . import EXT_cython


def construct_target_density(pm, approx, target_density_array=None, 
                             target_density_function=None):
    r"""Construct the target electron density, either from an input array, a 
    specified function or an input file.

    parameters
    ----------
    pm : object
        Parameters object 
    approx : string
        The approximation that is being used
    target_density_array : array_like
        1D array of the target electron density
    target_density_function : function
        Function specifying the form of the target electron density

    returns array_like
        1D array of the target electron density, indexed as 
        target_density[space_index]
    """
    if(target_density_array is not None):
        return target_density_array
    elif(target_density_function is not None):
        x_points = np.linspace(-pm.sys.xmax,pm.sys.xmax,pm.sys.grid)
        return target_density_function(x_points)
    else:
        name = 'gs_{}_den'.format(approx)
        results = rs.Results()
        return results.read(name, pm)
    

def calculate_density_error(pm, density, target_density):
    r"""Integrate the difference between the exact density and the target 
    density.

    .. math::

        \int|n(x) - n_{\mathrm{target}}(x)| dx

    parameters
    ----------
    pm : object
        Parameters object
    density : array_like
        1D array of the electron density, indexed as density[space_index]
    target_density : array_like
        1D array of the target electron density, indexed as 
        target_density[space_index]

    returns float
        The difference between the exact density and the target density, 
        density_error
    """
    density_difference = abs(density-target_density)
    density_error = np.sum(density_difference)*pm.sys.deltax

    return density_error


def calculate_deltav_ext(pm, density, target_density):
    r""" Iteratively correct the external potential.

    .. math::

        V_{\mathrm{ext}}(x) \rightarrow V_{\mathrm{ext}}(x) + 
        \mu[n^{p}(x)-n_{\mathrm{target}}^{p}(x)]

    parameters
    ----------
    pm : object
        Parameters object
    mu : float
        1st convergence parameter
    p : float
        2nd convergence parameter
    density : array_like
        1D array of the electron density, indexed as density[space_index]
    target_density : array_like
        1D array of the target electron density, indexed as
        target_density[space_index]

    returns array_like
        The correction to the external potential, deltav_ext
         
    """
    deltav_ext = pm.opt.mu*(density**pm.opt.p - target_density**pm.opt.p)

    return deltav_ext


def main(parameters, approx, target_density_array=None, 
         target_density_function=None):
    r""" Calculate the exact external potential for a specified electron
    density.
    
    parameters
    ----------
    parameters : object
        Parameters object
    approx : string
        The approximation that is being used
    target_density_array : array_like
        1D array of the target electron density
    target_density_function : function
        Function specifying the form of the target electron density

    returns object
        Results object 
    """
    # Array initialisations
    pm = parameters 
    string = 'OPT: constructing arrays'
    pm.sprint(string, 1, newline=True)
    pm.setup_space()

    # If the system contains one electron
    if(pm.sys.NE == 1):

        # Import EXT1
        from . import EXT1 as EXT

        # Construct the target density 
        target_density = construct_target_density(pm, approx, 
                         target_density_array, target_density_function)

        # Initial value of parameters 
        density_error = pm.opt.tol + 1.0
        density_error_old = np.copy(density_error) 
        v_ext_best = np.copy(pm.space.v_ext)
        run = 1

        # Calculate the external potential
        while(density_error > pm.opt.tol):

            # Print to screen
            if(run > 1):
                pm.sprint('', 1, newline=True)
            string = '---------------------------------------------------' + \
                     '-----------'
            pm.sprint(string, 1, newline=True)
            string = 'OPT: run = {}'.format(run)
            pm.sprint(string, 1, newline=True)
            string = '---------------------------------------------------' + \
                     '-----------'
            pm.sprint(string, 1, newline=True)

            # Construct the kinetic energy matrix
            K = EXT.construct_K(pm)

            # Construct the Hamiltonian matrix
            H = np.copy(K)
            H[0,:] += pm.space.v_ext[:]

            # Solve the Schrodinger equation
            energy, wavefunction = spla.eig_banded(H, lower=True, select='i',
                                   select_range=(0,0))

            # Normalise the wavefunction 
            normalisation = np.linalg.norm(wavefunction[:,0])*pm.sys.deltax**0.5
            wavefunction[:,0] /= normalisation
    
            # Calculate the electron density
            density = np.zeros(pm.sys.grid, dtype=np.float)
            density[:] = abs(wavefunction[:,0])**2
            if(run == 1):
                density_best = np.copy(density)

            # Calculate the error in the density
            density_error = calculate_density_error(pm, density, target_density)

            # Print to screen
            string = 'OPT: ground-state energy = {:.5f}'.format(energy[0])
            pm.sprint(string, 1, newline=True)
            string = 'OPT: density error = {:.5f}'.format(density_error)
            pm.sprint(string, 1, newline=True)

            # Ensure stable convergence 
            if(density_error < density_error_old):
                v_ext_best[:] = pm.space.v_ext[:]
                density_best[:] = density[:]
                if(abs(density_error - density_error_old) < 1e-8):
                    break
            else:
                pm.opt.mu = 0.5*pm.opt.mu

            # Correct the external potential 
            deltav_ext = calculate_deltav_ext(pm, density_best, target_density)
            pm.space.v_ext[:] = v_ext_best[:] + deltav_ext[:]
 
            # Iterate
            density_error_old = np.copy(density_error)
            run = run + 1

    # If the system contains two or three electrons
    else:

        # Import EXT2 or EXT3
        if(pm.sys.NE == 2):
            from . import EXT2 as EXT
        elif(pm.sys.NE == 3):
            from . import EXT3 as EXT

        # Construct the target density 
        target_density = construct_target_density(pm, approx, 
                         target_density_array, target_density_function)

        # Construct the reduction and expansion matrices
        reduction_matrix, expansion_matrix = (
                EXT.construct_antisymmetry_matrices(pm)) 

        # Generate the initial wavefunction
        wavefunction_reduced = EXT.initial_wavefunction(pm)

        # Initial value of parameters 
        density_error = pm.opt.tol + 1.0
        density_error_old = np.copy(density_error)
        v_ext_best = np.copy(pm.space.v_ext)
        run = 1

        # Calculate the external potential
        while(density_error > pm.opt.tol):
      
            # Print to screen
            if(run > 1):
                pm.sprint('', 1, newline=True)
            string = '---------------------------------------------------' + \
                     '-----------'
            pm.sprint(string, 1, newline=True)
            string = 'OPT: run = {}'.format(run)
            pm.sprint(string, 1, newline=True)
            string = '---------------------------------------------------' + \
                     '-----------'
            pm.sprint(string, 1, newline=True)

            # Construct the reduced form of the sparse matrices A and C 
            A_reduced = EXT.construct_A_reduced(pm, reduction_matrix, 
                        expansion_matrix, 0)
            C_reduced = -A_reduced + 2.0*reduction_matrix*sps.identity(
                        pm.sys.grid**pm.sys.NE, dtype=np.float)*expansion_matrix

            # Propagate through imaginary time
            energy, wavefunction = EXT.solve_imaginary_time(pm, A_reduced, 
                                   C_reduced, wavefunction_reduced, 
                                   reduction_matrix, expansion_matrix)

            # Calculate the electron density
            if(pm.sys.NE == 2):
                wavefunction_2D = wavefunction.reshape(pm.sys.grid, pm.sys.grid)
                density = EXT.calculate_density(pm, wavefunction_2D)
            elif(pm.sys.NE == 3):
                wavefunction_3D = wavefunction.reshape(pm.sys.grid, pm.sys.grid, pm.sys.grid)
                density = EXT.calculate_density(pm, wavefunction_3D)
            if(run == 1):
                density_best = np.copy(density)

            # Calculate the error in the density
            density_error = calculate_density_error(pm, density, target_density)

            # Print to screen
            string = 'OPT: density error = {:.5f}'.format(density_error)
            pm.sprint(string, 1, newline=True)

            # Ensure stable convergence 
            if(density_error < density_error_old):
                v_ext_best[:] = pm.space.v_ext[:]
                density_best[:] = density[:]
                if(abs(density_error - density_error_old) < 1e-8):
                    break
            else:
                pm.opt.mu = 0.5*pm.opt.mu

            # Correct the external potential 
            deltav_ext = calculate_deltav_ext(pm, density_best, target_density)
            pm.space.v_ext[:] = v_ext_best[:] + deltav_ext[:]

            # Set the initial wavefunction equal to the current wavefunction
            wavefunction_reduced = reduction_matrix*wavefunction
 
            # Iterate
            density_error_old = np.copy(density_error)
            run = run + 1 
                  
    # Print to screen 
    if(density_error < pm.opt.tol):
        string = 'OPT: The minimum tolerance has been met. Saving to file.'
        pm.sprint(string, 1, newline=True)
    else:  
        string = 'OPT: The minimum tolerance has not been met. Saving ' + \
                 'best guess to file.'
        pm.sprint(string, 1, newline=True)
    pm.sprint('', 1, newline=True)

    # Shift the external potential so that v_ext(x=0) = 0
    shift = v_ext_best[int((pm.sys.grid-1)/2)]
    v_ext_best[:] -= shift
    energy -= pm.sys.NE*shift
 
    # Save external potential, density, target density and energy 
    approxopt = approx + 'opt'
    results = rs.Results()
    results.add(v_ext_best,'gs_{}_vxt'.format(approxopt))
    results.add(density_best,'gs_{}_den'.format(approxopt))
    results.add(target_density,'gs_{}_tden'.format(approxopt))
    results.add(energy,'gs_{}_E'.format(approxopt))
    if(pm.run.save):
        results.save(pm)

    return results
