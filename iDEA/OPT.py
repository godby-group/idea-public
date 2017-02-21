"""Calculates the exact external potential and energy for a system of two or
three interacting electrons with a specified electron density.
"""


import numpy as np
import results as rs


def construct_target_density(pm, approx, x_points, target_density_array=None, 
                             target_density_function=None):
    r"""Construct the target electron density, either from an input array, a 
    specified function or an input file.

    parameters
    ----------
    pm : object
        Parameters object 
    approx : string
        The approximation that is being used
    x_points : array_like
        1D array containing the spatial grid points
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
	return target_density_function(x_points)
    else:
        name = 'gs_{}_den'.format(approx)
        results = rs.Results()
	return results.read(name, pm)
    

def calculate_density_error(pm, density, target_density):
    r"""Integrate the difference between the exact density and the target 
    density.

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
    density_error = np.linalg.norm(density-target_density)*pm.sys.deltax

    return density_error


def calculate_deltav_ext(pm, density, target_density):
    r""" Iteratively correct the external potential.

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
    pm = parameters

    # Import EXT2 or EXT3
    if(pm.sys.NE == 2):
        import EXT2 as EXT
    elif(pm.sys.NE == 3):
        import EXT3 as EXT
    else: 
        raise IOError("Must be either 2 or 3 electrons in the system.")

    # Array initialisations 
    wavefunction = np.zeros(pm.sys.grid**pm.sys.NE, dtype=np.cfloat)
    x_points = np.linspace(-pm.sys.xmax,pm.sys.xmax,pm.sys.grid)
    v_ext = pm.sys.v_ext(x_points)
    target_density = construct_target_density(pm, approx, x_points, 
                     target_density_array, target_density_function)
    x_points_tmp = np.linspace(0.0,2.0*pm.sys.xmax,pm.sys.grid)
    v_coulomb = 1.0/(pm.sys.acon + x_points_tmp)     

    # Construct the reduction and expansion matrices
    reduction_matrix, expansion_matrix = EXT.construct_antisymmetry_matrices(pm) 

    # Generate the initial wavefunction
    wavefunction_reduced = np.zeros(reduction_matrix.shape[0], dtype=np.cfloat)
    wavefunction_reduced = EXT.initial_wavefunction(pm, wavefunction_reduced)

    # Initial value of parameters 
    density_error = pm.opt.tol + 10.0
    density_error_old = np.copy(density_error) + 10.0
    run = 1

    # Calculate the external potential
    while(density_error > pm.opt.tol):
      
        # Print to screen
        string = '--------------------------------------------------------' + \
                 '------'
        pm.sprint(string, 1, newline=True)
        string = 'OPT: run = {}'.format(run)
        pm.sprint(string, 1, newline=True)
        string = '--------------------------------------------------------' + \
                 '------'
        pm.sprint(string, 1, newline=True)

        # Construct the reduced form of the sparse Matrix A
        A_reduced = EXT.construct_A_reduced(pm, reduction_matrix,
                    expansion_matrix, v_ext, v_coulomb, 0)

        # Propagate through imaginary time
        energy, wavefunction = EXT.solve_imaginary_time(pm, A_reduced, 
                               wavefunction_reduced, reduction_matrix, 
                               expansion_matrix)
      
        # Dispose of the reduced matrix
        del A_reduced

        # Calculate the electron density
        wavefunction_ND = EXT.wavefunction_converter(pm, wavefunction)
        density = EXT.calculate_density(pm, wavefunction_ND)

        # Calculate the error in the density
        density_error = calculate_density_error(pm, density, target_density)

        # Ensure stable convergence 
        if(density_error > density_error_old):
            pm.opt.mu = 0.5*pm.opt.mu

        # Print to screen
        string = 'OPT: density error = {:.5f}'.format(density_error)
        pm.sprint(string, 1, newline=True)
        pm.sprint('', 1, newline=True) 
   
        # Correct the external potential 
        deltav_ext = calculate_deltav_ext(pm, density, target_density)
        v_ext += deltav_ext

        # Iterate
        density_error_old = density_error
        run = run + 1

    v_ext -= deltav_ext

    # Save external potential, density and energy 
    approxopt = approx + 'opt'
    results = rs.Results()
    results.add(v_ext,'gs_{}_vxt'.format(approxopt))
    results.add(density,'gs_{}_den'.format(approxopt))
    results.add(energy,'gs_{}_E'.format(approxopt))
    if(pm.run.save):
        results.save(pm)