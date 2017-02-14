"""Calculates the exact (Dobson) electron localisation function (ELF) for a two
 or three electron system using the many-body wavefunction. 
"""


import numpy as np


def calculate_density(pm,wavefunction_ND):
    r"""Calculates the electron density from the many-body wavefunction.

    parameters
    ----------
    pm : object
        Parameters object
    wavefunction_ND : array_like
        ND array of the N-electron wavefunction, indexed as 
        wavefunction_ND[{space_index_i}]

    returns array_like
        1D array of the electron density, indexed as density[space_index]
    """
    if(pm.sys.NE == 2):
        density = 2.0*np.sum(np.abs(wavefunction_ND)**2, axis=1, 
                  dtype=np.float)*pm.sys.deltax
    elif(pm.sys.NE == 3):
        density = 3.0*np.sum(np.sum(np.abs(wavefunction_ND)**2, axis=1, 
                  dtype=np.float), axis=1)*pm.sys.deltax**2    
    else:
        raise ValueError('ELF does not support systems containing {} electrons'
                         .format(pm.sys.NE))

    return density


def calculate_pair_density(pm,wavefunction_ND):
    r"""Calculates the electron pair density from the many-body wavefunction.

    parameters
    ----------
    pm : object
        Parameters object
    wavefunction_ND : array_like
        ND array of the N-electron wavefunction, indexed as 
        wavefunction_ND[{space_index_i}]
 
    returns array_like
        2D array of the electron pair density, indexed as 
        pair_density[space_index_1, space_index_2]
    """
    if(pm.sys.NE == 2):
        pair_density = pm.sys.NE*(pm.sys.NE-1)*(np.abs(wavefunction_ND))**2
    elif(pm.sys.NE == 3):
        pair_density = pm.sys.NE*(pm.sys.NE-1)*np.sum(np.abs(wavefunction_ND
                       )**2, axis=2, dtype=np.float)*pm.sys.deltax
    else:
        raise ValueError('ELF does not support systems containing {} electrons'
                         .format(pm.sys.NE))
        
    return pair_density


def calculate_d_sigma(pm,density,pair_density):
    r"""Calculates D_sigma from the electron density and the electron pair 
    density.

    parameters
    ----------
    pm : object
        Parameters object
    density : array_like
        1D array of the electron density, indexed as density[space_index]
    pair_density : array_like
        2D array of the electron pair density, indexed as 
        pair_density[space_index_1, space_index_2]

    returns array_like
        1D array of D_sigma, indexed as d_sigma[space_index]
    """
    laplacian = np.gradient(np.array(np.gradient(pair_density, pm.sys.deltax, 
                axis=1)), pm.sys.deltax, axis=1)
    d_sigma = np.zeros(pm.sys.grid, dtype=np.float)
    for i in range(pm.sys.grid):
        d_sigma[i] = laplacian[i,i]/(2.0*density[i])
    
    return d_sigma


def elf_dobson(pm,density,d_sigma):
    r"""Calculate the ELF from the electron density and the electron pair 
    density.

    parameters
    ----------
    pm : object
        Parameters object
    density : array_like
        1D array of the electron density, indexed as density[space_index]
    d_sigma : array_like 
        1D array of D_sigma, indexed as d_sigma[space_index]

    returns array_like
        1D array of the ELF, indexed as elf[space_index]
    """
    d_sigma_h = (np.pi**2)*(density**3)/6.0
    elf = 1.0/(1.0 + (d_sigma/d_sigma_h)**2)

    return elf


def main(parameters,wavefunction_ND):
    r"""Calculates the ELF from the many-body wavefunction.

    parameters
    ----------
    parameters : object
        Parameters object
    wavefunction_ND : array_like
        ND array of the N-particle wavefunction, indexed as 
        wavefunction_ND[{space_index_i}] 

    returns array_like
        1D array of the exact ELF, indexed as elf[space_index]
    """
    pm = parameters

    density = calculate_density(pm,wavefunction_ND)
    pair_density = calculate_pair_density(pm,wavefunction_ND)
    d_sigma = calculate_d_sigma(pm,density,pair_density)
    elf = elf_dobson(pm,density,d_sigma)

    return elf

