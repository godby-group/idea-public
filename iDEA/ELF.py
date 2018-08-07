"""Calculates the exact (Dobson) electron localisation function (ELF) for a two or three electron 
system using the many-electron wavefunction. 
"""

from __future__ import division            
from __future__ import print_function      
from __future__ import absolute_import                   

import numpy as np


def calculate_density(pm, wavefunction_ND):
    r"""Calculates the electron density from the many-electron wavefunction.

    .. math:: 
   
        2e: n(x) = 2 \int_{-x_{\mathrm{max}}}^{x_{\mathrm{max}}} |\Psi(x,x_{2})|^2 dx_{2} \\

        3e: n(x) = 3 \int_{-x_{\mathrm{max}}}^{x_{\mathrm{max}}} |\Psi(x,x_{2},x_{3})|^{2} dx_{2} \ dx_{3}

    parameters
    ----------
    pm : object
        Parameters object
    wavefunction_ND : array_like
        ND array of the N-electron wavefunction, indexed as wavefunction_ND[{space_index_i}]

    returns array_like
        1D array of the electron density, indexed as density[space_index]
    """
    # Two electrons
    if(pm.sys.NE == 2):
        mod_wavefunction_2D = np.absolute(wavefunction_ND)**2
        density = 2.0*np.sum(mod_wavefunction_2D, axis=1, dtype=np.float)*pm.space.delta

    # Three electrons
    elif(pm.sys.NE == 3):
        mod_wavefunction_3D = np.absolute(wavefunction_ND)**2
        density = 3.0*np.sum(np.sum(mod_wavefunction_3D, axis=1, dtype=np.float), axis=1)*pm.space.delta**2

    # Raise error
    else:
        raise ValueError('ELF does not support systems containing {} electrons'.format(pm.sys.NE))

    return density


def calculate_pair_density(pm, wavefunction_ND):
    r"""Calculates the electron pair density from the many-electron wavefunction.

    .. math:
 
        2e: n_{2}(x,x') = 2 |\Psi(x,x_{2})|^2 

        3e: n_{2}(x,x') = 6 \int_{-x_{\mathrm{max}}}^{x_{\mathrm{max}}} |\Psi(x,x_{2},x_{3})|^2 dx_{3}

    parameters
    ----------
    pm : object
        Parameters object
    wavefunction_ND : array_like
        ND array of the N-electron wavefunction, indexed as wavefunction_ND[{space_index_i}]
 
    returns array_like
        2D array of the electron pair density, indexed as pair_density[space_index_1, space_index_2]
    """
    # Two electrons
    if(pm.sys.NE == 2):
        pair_density = pm.sys.NE*(pm.sys.NE-1)*(np.absolute(wavefunction_ND))**2

    # Three electrons
    elif(pm.sys.NE == 3):
        pair_density = pm.sys.NE*(pm.sys.NE-1)*np.sum(np.absolute(wavefunction_ND)**2, axis=2, dtype=np.float)*pm.space.delta
    
    # Raise error 
    else:
        raise ValueError('ELF does not support systems containing {} electrons'.format(pm.sys.NE))
        
    return pair_density


def calculate_D_sigma(pm, density, pair_density):
    r"""Calculates D_sigma from the electron density and the electron pair density.

    .. math ::

       D_{\sigma}(x) = \frac{\Big[\nabla^{2} _{x'} n_{2}(x,x') \Big]_{x'=x}}{2n(x)}

    parameters
    ----------
    pm : object
        Parameters object
    density : array_like
        1D array of the electron density, indexed as density[space_index]
    pair_density : array_like
        2D array of the electron pair density, indexed as pair_density[space_index_1, space_index_2]

    returns array_like
        1D array of D_sigma, indexed as D_sigma[space_index]
    """
    # Laplacian of the electron pair density 
    laplacian = np.gradient(np.array(np.gradient(pair_density, pm.space.delta, axis=1)), pm.space.delta, axis=1)

    # Calculate D_sigma
    D_sigma = np.zeros(pm.space.npt, dtype=np.float)
    for j in range(pm.space.npt):
        D_sigma[j] = laplacian[j,j]/(2.0*density[j])
    
    return D_sigma


def elf_dobson(pm, density, D_sigma):
    r"""Calculate the ELF from the electron density and the electron pair density.

    .. math ::

        \mathrm{ELF}(x) = \frac{1}{1 + \Big(\frac{D_{\sigma}(x)}{D_{\sigma, \mathrm{H}}(x)}\Big)^{2}} \\

        D_{\sigma, \mathrm{H}}(x) = \frac{\pi^{2}}{6} \big[n(x)\big]^{3}

    parameters
    ----------
    pm : object
        Parameters object
    density : array_like
        1D array of the electron density, indexed as density[space_index]
    D_sigma : array_like 
        1D array of D_sigma, indexed as D_sigma[space_index]

    returns array_like
        1D array of the ELF, indexed as elf[space_index]
    """
    D_sigma_h = (np.pi**2)*(density**3)/6.0
    elf = 1.0/(1.0 + (D_sigma/D_sigma_h)**2)

    return elf


def main(parameters, wavefunction_ND, density=None):
    r"""Calculates the ELF from the many-electron wavefunction.

    parameters
    ----------
    parameters : object
        Parameters object
    wavefunction_ND : array_like
        ND array of the N-electron wavefunction, indexed as wavefunction_ND[{space_index_i}] 
    density : array_like
        1D array of the electron density, indexed as density[space_index]

    returns array_like
        1D array of the exact ELF, indexed as elf[space_index]
    """
    pm = parameters

    if((density == None).any()):
        density = calculate_density(pm, wavefunction_ND)
    pair_density = calculate_pair_density(pm, wavefunction_ND)
    D_sigma = calculate_D_sigma(pm, density, pair_density)
    elf = elf_dobson(pm, density, D_sigma)

    return elf

