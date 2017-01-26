"""Calculates the exact (Dobson) electron localisation function (ELF) for a two or three electron system using the many-body wavefunction. 
"""


import numpy as np


def Den(pm,PsiND):
    r"""Calculate the electron density from the many-body wavefunction

    parameters
    ----------
    pm : object
        Parameters object
    PsiND : array_like
        ND array of N-particle wavefunction, indexed as PsiND[{space_index_i}]

    returns array_like
        1D array of electron density, indexed as n[space_index]
    """
    if(pm.sys.NE == 2):
        n = 2.0*np.sum(np.abs(PsiND)**2, axis=1, dtype=np.float)*pm.sys.deltax
    elif(pm.sys.NE == 3):
        n = 3.0*np.sum(np.sum(np.abs(PsiND)**2, axis=1, dtype=np.float), axis=1)*pm.sys.deltax**2    
    else:
        raise ValueError('ELF does not support systems containing {} electrons'.format(pm.sys.NE))

    return n 


def PairDen(pm,PsiND):
    r"""Calculate the electron pair density from the many-body wavefunction

    parameters
    ----------
    pm : object
        Parameters object
    PsiND : array_like
        ND array of N-particle wavefunction, indexed as PsiND[{space_index_i}]
 
    returns array_like
        2D array of electron pair density, indexed as n2[space_index_1, space_index_2]
    """
    if(pm.sys.NE == 2):
        n2 = pm.sys.NE*(pm.sys.NE-1)*(np.abs(PsiND))**2
    elif(pm.sys.NE == 3):
        n2 = pm.sys.NE*(pm.sys.NE-1)*np.sum(np.abs(PsiND)**2, axis=2, dtype=np.float)*pm.sys.deltax
    else:
        raise ValueError('ELF does not support systems containing {} electrons'.format(pm.sys.NE))
        
    return n2


def DSigma(pm,n,n2):
    r"""Calculate DSigma from the electron density and the electron pair density

    parameters
    ----------
    pm : object
        Parameters object
    n : array_like
        1D array of electron density, indexed as n[space_index]
    n2 : array_like
        2D array of electron pair density, indexed as n2[space_index_1, space_index_2]

    returns array_like
        1D array of DSigma, indexed as Ds[space_index]
    """
    Laplacian = np.gradient(np.array(np.gradient(n2, pm.sys.deltax, axis=1)), pm.sys.deltax, axis=1)
    Ds = np.zeros(pm.sys.grid, dtype=np.float)
    for i in range(pm.sys.grid):
        Ds[i] = Laplacian[i,i]/(2.0*n[i])
    
    return Ds


def ElfDobson(pm,n,Ds):
    r"""Calculate ELF from the electron density and the electron pair density

    parameters
    ----------
    pm : object
        Parameters object
    n : array_like
        1D array of electron density, indexed as n[space_index]
    Ds : array_like 
        1D array of DSigma, indexed as Ds[space_index]

    returns array_like
        1D array of ELF, indexed as ELF[space_index]
    """
    Dsh = (np.pi**2)*(n**3)/6.0
    ELF = 1.0/(1.0 + (Ds/Dsh)**2)

    return ELF


def main(parameters,PsiND):
    r"""Calculates ELF from the many-body wavefunction

    parameters
    ----------
    parameters : object
        Parameters object
    PsiND : array_like
        ND array of N-particle wavefunction, indexed as PsiND[{space_index_i}] 

    returns array_like
        1D array of exact ELF, indexed as ELF[space_index]
    """
    pm = parameters

    n = Den(pm,PsiND)
    n2 = PairDen(pm,PsiND)
    Ds = DSigma(pm,n,n2)
    ELF = ElfDobson(pm,n,Ds)

    return ELF

