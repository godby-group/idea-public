"""Uses the [adiabatic] local density approximation ([A]LDA) to calculate the [time-dependent] 
electron density [and current] for a system of N electrons.

Computes approximations to V_KS, V_H, V_xc using the LDA self-consistently. For ground state 
calculations the code outputs the LDA orbitals and energies of the system, the ground-state 
charge density and Kohn-Sham potential. For time dependent calculations the code also outputs 
the time-dependent charge and current densities and the time-dependent Kohn-Sham potential.

Note: Uses the LDAs developed in [Entwistle2018]_ from finite slab systems and the HEG, 
in one dimension.
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import copy
import numpy as np
import scipy as sp
import scipy.sparse as sps
import scipy.linalg as spla
import scipy.sparse.linalg as spsla

from . import LDA_parameters
from . import RE_cython
from . import results as rs
from . import mix
from . import minimize


def groundstate(pm, H):
    r"""Calculates the ground-state of the system for a given potential.

    .. math:: 
    
        \hat{H} \phi_{j} = \varepsilon_{j} \phi_{j}

    parameters
    ----------
    pm : object
        Parameters object
    H : array_like
        2D array of the Hamiltonian matrix in band form, indexed as H[band,space_index]

    returns
    -------
    density : array_like
        1D array of the electron density, indexed as density[space_index]
    orbitals : array_like
        2D array of the Kohn-Sham orbitals, index as orbitals[space_index,orbital_number]
    eigenvalues : array_like
        1D array of the Kohn-Sham eigenvalues, indexed as eigenvalues[orbital_number]
    """
    # Solve the Kohn-Sham equations
    eigenvalues, orbitals = spla.eig_banded(H, lower=True)
 
    # Normalise the orbitals
    orbitals /= np.sqrt(pm.space.delta)

    # Calculate the electron density
    density = electron_density(pm, orbitals)

    return density, orbitals, eigenvalues


def electron_density(pm, orbitals):
    r"""Calculates the electron density from the set of orbitals.

    .. math:: 

        n(x) = \sum_{j=1}^{N}|\phi_{j}(x)|^{2}

    parameters
    ----------
    pm : object
        Parameters object
    orbitals : array_like
        2D array of the Kohn-Sham orbitals, index as orbitals[space_index,orbital_number]

    returns
    -------
    density : array_like
        1D array of the electron density, indexed as density[space_index]
    """
    density = np.sum(np.absolute(orbitals[:,:pm.sys.NE])**2, axis=1)

    return density


def ks_potential(pm, density, perturbation=False):
    r"""Calculates the Kohn-Sham potential from the electron density.

    .. math::

        V_{\mathrm{KS}} = V_{\mathrm{ext}} + V_{\mathrm{H}} + V_{\mathrm{xc}}

    parameters
    ----------
    pm : object
        Parameters object
    density : array_like
        1D array of the electron density, indexed as density[space_index]
    perturbation: bool
      - True: Perturbed external potential
      - False: Unperturbed external potential

    returns
    -------
    v_ks : array_like
        1D array of the Kohn-Sham potential, indexed as v_ks[space_index]
    """
    v_ks = pm.space.v_ext + hartree_potential(pm, density) + xc_potential(pm, density)
    if perturbation:
        v_ks += pm.space.v_pert

    return v_ks


def banded_to_full(pm, H):
    r"""Converts the Hamiltonian matrix in band form to the full matrix.

    parameters
    ----------
    pm : object
        Parameters object
    H : array_like
        2D array of the Hamiltonian matrix in band form, indexed as H[band,space_index]

    returns
    -------
    H_full : array_like
        2D array of the Hamiltonian matrix in full form, indexed as H_full[space_index,space_index]
    """
    # Stencil used
    sd = pm.space.second_derivative_band
    nbnd = len(sd)

    # Add the band elements to the full matrix
    H_full = np.zeros((pm.space.npt,pm.space.npt), dtype=np.float)
    for ioff in range(nbnd):
        d = np.arange(pm.space.npt-ioff)
        H_full[d,d+ioff] = H[ioff,d]
        H_full[d+ioff,d] = H[ioff,d]

    return H_full


def kinetic(pm):
    r"""Stores the band elements of the kinetic energy matrix in lower form. The kinetic energy matrix 
    is constructed using a three-point, five-point or seven-point stencil. This yields an NxN band 
    matrix (where N is the number of grid points). For example with N=6 and a three-point stencil:
   
    .. math::

        K = -\frac{1}{2} \frac{d^2}{dx^2}=                
        -\frac{1}{2} \begin{pmatrix}                      
        -2 & 1 & 0 & 0 & 0 & 0 \\                         
        1 & -2 & 1 & 0 & 0 & 0 \\                         
        0 & 1 & -2 & 1 & 0 & 0 \\                         
        0 & 0 & 1 & -2 & 1 & 0 \\                         
        0 & 0 & 0 & 1 & -2 & 1 \\                         
        0 & 0 & 0 & 0 & 1 & -2                            
        \end{pmatrix}                                     
        \frac{1}{\delta x^2}                              
        = [\frac{1}{\delta x^2},-\frac{1}{2 \delta x^2}]  

    parameters
    ----------
    pm : object
        Parameters object

    returns array_like
        2D array containing the band elements of the kinetic energy matrix, indexed as 
        K[band,space_index]
    """
    # Stencil to use
    sd = pm.space.second_derivative_band
    nbnd = len(sd)

    # Band elements
    K = np.zeros((nbnd, pm.space.npt), dtype=np.float)
    for i in range(nbnd):
        K[i,:] = -0.5 * sd[i]

    return K


def hamiltonian(pm, v_ks=None, orbitals=None, perturbation=False):
    r"""Constructs the Hamiltonian matrix in band form for a given Kohn-Sham potential.

    .. math::

        \hat{H} = \hat{K} + \hat{V}_{\mathrm{KS}}

    parameters
    ----------
    pm : object
        Parameters object
    v_ks : array_like
        1D array of the Kohn-Sham potential, indexed as v_ks[space_index]
    orbitals : array_like
        2D array of the Kohn-Sham orbitals, index as orbitals[space_index,orbital_number]
    perturbation: bool
      - True: Perturbed external potential
      - False: Unperturbed external potential

    returns
    -------
    H : array_like
         2D array of the Hamiltonian matrix in band form, indexed as H[band,space_index]
    """
    # Kinetic energy matrix
    H = kinetic(pm)

    # Calculate the Kohn-Sham potential from the orbitals
    if orbitals is not None:
        density = electron_density(pm, orbitals)
        if perturbation:
            v_ks = ks_potential(pm, density, perturbation=True) 
        else:
            v_ks = ks_potential(pm, density)

    # Add the Kohn-Sham potential to the Hamiltonian
    H[0,:] += v_ks

    return H


def hartree_potential(pm, density):
    r"""Calculates the Hartree potential for a given electron density.

    .. math::

        V_{\mathrm{H}}(x) = \int U(x,x') n(x') dx'

    parameters
    ----------
    pm : object
        Parameters object
    density : array_like
        1D array of the electron density, indexed as density[space_index]

    returns array_like
        1D array of the Hartree potential, indexed as v_h[space_index]
    """
    v_h = np.dot(pm.space.v_int,density)*pm.space.delta

    return v_h


def hartree_energy(pm, v_h, density):
    r"""Calculates the Hartree energy of the ground-state system.

    .. math::

        E_{\mathrm{H}}[n] = \frac{1}{2} \int \int U(x,x') n(x) n(x') dx dx'
        = \frac{1}{2} \int V_{\mathrm{H}}(x) n(x) dx

    parameters
    ----------
    pm : object
        Parameters object
    v_h : array_like
        1D array of the ground-state Hartree potential, indexed as v_h[space_index]
    density : array_like
        1D array of the ground-state electron density, indexed as density[space_index]

    returns float
        The Hartree energy of the ground-state system
    """
    E_h = 0.5*np.dot(v_h,density)*pm.space.delta

    return E_h


def xc_energy(pm, n, separate=False):
    r"""LDA approximation for the exchange-correlation energy. Uses the LDAs developed in 
    [Entwistle et al. 2018] from finite slab systems and the HEG.

    .. math ::

        E_{\mathrm{xc}}^{\mathrm{LDA}}[n] = \int \varepsilon_{\mathrm{xc}}(n) n(x) dx

    parameters
    ----------
    pm : object
        Parameters object
    n : array_like
        1D array of the electron density, indexed as n[space_index]
    separate: bool
      - True: Split the HEG exchange-correlation energy into separate exchange and correlation terms
      - False: Just return the exchange-correlation energy 

    returns float
        Exchange-correlation energy
    """
    NE = pm.lda.NE

    # Finite LDAs
    if NE != 'heg':
        p = LDA_parameters.exc_lda[NE]
        e_xc = (p['a'] + p['b']*n + p['c']*n**2 + p['d']*n**3 + p['e']*n**4 + p['f']*n**5)*n**p['g']
    
    # HEG LDA
    else:
        p = LDA_parameters.ex_lda[NE]
        q = LDA_parameters.ec_lda[NE]
        e_x = np.zeros(pm.space.npt, dtype=np.float)
        e_c = np.copy(e_x)
        for j in range(pm.space.npt):
            if(n[j] != 0.0):

                # Exchange energy per electron
                e_x[j] = (p['a'] + p['b']*n[j] + p['c']*n[j]**2 + p['d']*n[j]**3 + p['e']*n[j]**4 + p['f']*n[j]**5)*n[j]**p['g']

                # Correlation energy per electron
                r_s = 0.5/n[j]
                e_c[j] = -((q['a']*r_s + q['e']*r_s**2)/(1.0 + q['b']*r_s + q['c']*r_s**2 + q['d']*r_s**3))*np.log(1.0 + \
                         q['f']*r_s + q['g']*r_s**2)/q['f']

        # Exchange-correlation energy per electron
        e_xc = e_x + e_c

    # Exchange-correlation energy
    E_xc = np.dot(e_xc, n)*pm.space.delta

    # Separate exchange and correlation contributions
    if separate == True:
        E_x = np.dot(e_x, n)*pm.space.delta
        E_c = np.dot(e_c, n)*pm.space.delta
        return E_xc, E_x, E_c
    else:
        return E_xc


def xc_potential(pm, n, separate=False):
    r"""LDA approximation for the exchange-correlation potential. Uses the LDAs developed in 
    [Entwistle et al. 2018] from finite slab systems and the HEG.

    .. math ::

        V_{\mathrm{xc}}^{\mathrm{LDA}}(x) = \frac{\delta E_{\mathrm{xc}}^{\mathrm{LDA}}[n]}{\delta n(x)}
        = \varepsilon_{\mathrm{xc}}(n(x)) + n(x)\frac{d\varepsilon_{\mathrm{xc}}}{dn} \bigg|_{n(x)}

    parameters
    ----------
    pm : object
        Parameters object
    n : array_like
        1D array of the electron density, indexed as n[space_index]
    separate: bool
      - True: Split the HEG exchange-correlation potential into separate exchange and correlation terms
      - False: Just return the exchange-correlation potential 

    returns array_like
        1D array of the exchange-correlation potential, indexed as v_xc[space_index]
    """
    NE = pm.lda.NE

    # Finite LDAs
    if NE != 'heg':
        p = LDA_parameters.vxc_lda[NE]
        v_xc = (p['a'] + p['b']*n + p['c']*n**2 + p['d']*n**3 + p['e']*n**4 + p['f']*n**5)*n**p['g']

    # HEG LDA
    else:
        p = LDA_parameters.vx_lda[NE]
        q = LDA_parameters.ec_lda[NE]
        v_x = np.zeros(pm.space.npt, dtype=np.float)
        v_c = np.copy(v_x)
        for j in range(pm.space.npt):
            if n[j] != 0.0:

                # Exchange potential
                v_x[j] = (p['a'] + p['b']*n[j] + p['c']*n[j]**2 + \
                         p['d']*n[j]**3 + p['e']*n[j]**4 + \
                         p['f']*n[j]**5)*n[j]**p['g']

                # Correlation potential
                r_s = 0.5/n[j]
                energy = -((q['a']*r_s + q['e']*r_s**2)/(1.0 + q['b']*r_s + q['c']*r_s**2 + q['d']*r_s**3))*\
                         np.log(1.0 + q['f']*r_s + q['g']*r_s**2)/q['f']
                derivative = ((r_s*(q['a'] + q['e']*r_s)*(q['b'] + r_s*(2.0*q['c'] + 3.0*q['d']*r_s))*np.log(1.0 + \
                             q['f']*r_s + q['g']*(r_s**2)) - (r_s*(q['a'] + q['e']*r_s)*(q['f'] + 2.0*q['g']*r_s)*\
                             (q['b']*r_s + q['c']*(r_s**2) + q['d']*(r_s**3) + 1.0)/(q['f']*r_s + q['g']*(r_s**2) + \
                             1.0)) - ((q['a'] + 2.0*q['e']*r_s)*(q['b']*r_s + q['c']*(r_s**2) + q['d']*(r_s**3) + 1.0)*\
                             np.log(1.0 + q['f']*r_s + q['g']*(r_s**2))))/(q['f']*(q['b']*r_s + q['c']*(r_s**2) + \
                             q['d']*(r_s**3) + 1.0)**2))

                v_c[j] = energy - r_s*derivative
 
        # Exchange-correlation potential
        v_xc = v_x + v_c

    if separate == True:
        return v_xc, v_x, v_c
    else:
        return v_xc


def DXC(pm, n): 
    r"""Calculates the derivative of the exchange-correlation potential, necessary for the RPA 
    preconditioner.

    parameters
    ----------
    pm : object
        Parameters object
    n : array_like
        1D array of the electron density, indexed as n[space_index]

    returns array_like
        1D array of the derivative of the exchange-correlation potential, indexed as D_xc[space_index]
    """
    NE = pm.lda.NE

    # Currently only the finite LDAs can be used
    if NE != 'heg':
        p = LDA_parameters.dlda[NE]
        D_xc = (p['a'] + n*(p['b'] + n*(p['c'] + n*(p['d'] + n*(p['e'] + n*p['f'])))))*(n**p['g'])
    else: 
        raise IOError("Currently the HEG LDA is not implemented for the RPA preconditioner.")

    return D_xc 


def total_energy_eigv(pm, eigenvalues, orbitals=None, density=None, v_h=None, v_xc=None):
    r"""Calculates the total energy from the Kohn-Sham eigenvalues.

    .. math ::

        E[n] = \sum_{j=1}^{N} \varepsilon_j + E_{xc}[n] - E_H[n] - \int n(x) V_{xc}(x)dx

    parameters
    ----------
    pm : object
        Parameters object
    eigenvalues : array_like
        1D array of the Kohn-Sham eigenvalues, indexed as eigenvalues[orbital_number]
    orbitals : array_like
        2D array of the Kohn-Sham orbitals, index as orbitals[space_index,orbital_number]
    density : array_like
        1D array of the electron density, indexed as density[space_index]
    v_h : array_like
        1D array of the Hartree potential, indexed as v_h[space_index]
    v_xc : array_like
        1D array of the exchange-correlation potential, indexed as v_xc[space_index]

    returns float
        Total energy
    """
    # Quantities needed to calculate the total energy
    if density is None:
        if orbitals is None:
            raise ValueError("Need to specify either density or orbitals")
        else:
            density = electron_density(pm, orbitals)
    if v_h is None:
        v_h = hartree_potential(pm, density)
    if v_xc is None:
        v_xc = xc_potential(pm, density)

    # Kohn-Sham eigenvalues
    E = 0.0
    for j in range(pm.sys.NE):
        E += eigenvalues[j]

    # Hartree Energy
    E -= hartree_energy(pm, v_h, density)

    # Exchange-correlation potential term
    E -= np.dot(density, v_xc)*pm.space.delta
 
    # Exchange-correlation energy
    E += xc_energy(pm, density)

    return E.real


def total_energy_eigf(pm, orbitals, density=None, v_h=None):
    r"""Calculates the total energy from the Kohn-Sham orbitals.

    .. math ::

        E[n] = \sum_{j=1}^{N} \langle \phi_{j} | K | \phi_{j} \rangle + E_H[n] + E_{xc}[n] 
        + \int n(x) V_{\mathrm{ext}}(x)dx

    parameters
    ----------
    pm : object
        Parameters object
    orbitals : array_like
        2D array of the Kohn-Sham orbitals, index as orbitals[space_index,orbital_number]
    density : array_like
        1D array of the electron density, indexed as density[space_index]
    v_h : array_like
        1D array of the Hartree potential, indexed as v_h[space_index]

    returns float
        Total energy
    """
    # Quantities needed to calculate the total energy
    if density is None:
        density = electron_density(pm, orbitals)
    if v_h is None:
        v_h = hartree_potential(pm, density)

    # Kinetic energy
    E = 0.0
    E += kinetic_energy(pm, orbitals)

    # Hartree energy
    E += hartree_energy(pm, v_h, density)

    # Exchange-correlation energy
    E += xc_energy(pm, density)

    # External potential term
    E += np.dot(pm.space.v_ext, density)*pm.space.delta

    return E.real


def kinetic_energy(pm, orbitals):
    r"""Calculates the kinetic energy from the Kohn-Sham orbitals.

    .. math ::

        T_{s}[n] = \sum_{j=1}^{N} \langle \phi_{j} | K | \phi_{j} \rangle

    parameters
    ----------
    pm : object
        Parameters object
    orbitals : array_like
        2D array of the Kohn-Sham orbitals, index as orbitals[space_index,orbital_number]

    """
    # Kinetic energy matrix
    sd = pm.space.second_derivative
    sd_ind = pm.space.second_derivative_indices
    K = -0.5*sps.diags(sd, sd_ind, shape=(pm.space.npt, pm.space.npt), dtype=np.float, format='csr')

    # Kinetic energy of each occupied orbital
    occ = orbitals[:,:pm.sys.NE]
    eigenvalues = (occ.conj() * K.dot(occ)).sum(0)*pm.space.delta

    return np.sum(eigenvalues)


def calculate_current_density(pm, density):
    r"""Calculates the current density from the time-dependent electron density by solving the 
    continuity equation.

    .. math:: 

        \frac{\partial n}{\partial t} + \frac{\partial j}{\partial x} = 0

    parameters
    ----------
    pm : object
        Parameters object
    density : array_like
        2D array of the time-dependent density, indexed as density[time_index,space_index]

    returns array_like
        2D array of the current density, indexed as current_density[time_index,space_index]
    """
    pm.sprint('', 1)
    string = 'LDA: calculating current density'
    pm.sprint(string, 1)
    current_density = np.zeros((pm.sys.imax,pm.space.npt), dtype=np.float)
    for i in range(1, pm.sys.imax):
         string = 'LDA: t = {:.5f}'.format(i*pm.sys.deltat)
         pm.sprint(string, 1, newline=False)
         J = np.zeros(pm.space.npt, dtype=np.float)
         J = RE_cython.continuity_eqn(pm, density[i,:], density[i-1,:])
         current_density[i,:] = J[:]
    pm.sprint('', 1)

    return current_density


def crank_nicolson_step(pm, orbitals, H_full):
    r"""Solves Crank Nicolson Equation

    .. math::

        \left(I + i\frac{dt}{2} H\right) \Psi(x,t+dt) = \left(I - i \frac{dt}{2} H\right) \Psi(x,t)

    parameters
    ----------
    pm : object
        Parameters object
    orbitals : array_like
        2D array of the Kohn-Sham orbitals, index as orbitals[space_index,orbital_number]
    H_full : array_like
        2D array of the Hamiltonian matrix in full form, indexed as H_full[space_index,space_index]

    returns
    """
    # Construct matrices
    dH = 0.5j*pm.sys.deltat*H_full
    identity = np.identity(pm.space.npt, dtype=np.cfloat)
    A = identity + dH
    Abar = identity - dH
 
    # Solve for all single-particle states at once
    RHS = np.dot(Abar, orbitals[:, :pm.sys.NE])
    orbitals_new = spla.solve(A, RHS)

    return orbitals_new


def main(parameters):
    r"""Performs LDA calculation

    parameters
    ----------
    parameters : object
        Parameters object

    returns object
        Results object
    """
    # Array initialisations 
    pm = parameters
    string = 'LDA: constructing arrays'
    pm.sprint(string, 1) 
    pm.setup_space()

    # Take external potential as the initial guess to the Kohn-Sham potential
    H = hamiltonian(pm, v_ks=pm.space.v_ext)
    n_inp, orbitals, eigenvalues = groundstate(pm, H)
    E = total_energy_eigv(pm, eigenvalues=eigenvalues, density=n_inp)

    # Need n_inp and n_out to start mixing
    H = hamiltonian(pm, v_ks=ks_potential(pm, n_inp))
    n_out, orbitals_out, eigenvalues_out = groundstate(pm, H)

    # Mixing scheme
    if pm.lda.scf_type == 'pulay':
        mixer = mix.PulayMixer(pm, order=pm.lda.pulay_order, preconditioner=pm.lda.pulay_preconditioner)
    elif pm.lda.scf_type == 'cg':
        minimizer = minimize.CGMinimizer(pm, total_energy_eigf)
    elif pm.lda.scf_type == 'mixh':
        minimizer = minimize.DiagMinimizer(pm, total_energy_eigf)
        H_mix = copy.copy(H)

    # Find the self-consistent solution
    iteration = 1
    converged = False
    while (not converged) and iteration <= pm.lda.max_iter:
        E_old = E

        # Conjugate-gradient minimization starts with orbitals, H[orbitals]
        if pm.lda.scf_type ==  'cg':

            orbitals = minimizer.step(orbitals, banded_to_full(pm, H))
            n_inp = electron_density(pm, orbitals)

            # Calculate total energy at n_inp
            E = total_energy_eigf(pm, orbitals=orbitals, density=n_inp)

        # Minimization that mixes Hamiltonian directly starts with n_inp, H[n_inp]
        elif pm.lda.scf_type == 'mixh':

            n_tmp, orbitals_tmp, eigenvalues_tmp = groundstate(pm,H_mix)
            H_tmp = hamiltonian(pm, v_ks=ks_potential(pm, n_tmp))

            H_mix = minimizer.h_step(H_mix, H_tmp)
            n_inp, orbitals_inp, eigenvalues_inp = groundstate(pm,H_mix)

            # Calculate total energy at n_inp
            E = total_energy_eigv(pm, eigenvalues=eigenvalues_inp, density=n_inp)

        # Mixing schemes starting with n_inp, n_out (Pulay, linear or none)
        else:

            # Calculate new n_inp
            if pm.lda.scf_type == 'pulay':
                n_inp = mixer.mix(n_inp, n_out, eigenvalues_out, orbitals_out.T)
            elif pm.lda.scf_type == 'linear':
                n_inp = (1-pm.lda.mix)*n_inp + pm.lda.mix*n_out
            else:
                n_inp = n_out

            # Calculate total energy at n_inp
            E = total_energy_eigv(pm, eigenvalues=eigenvalues_out, density=n_inp)

        # Calculate new Kohn-Sham potential and update the Hamiltonian
        v_ks = ks_potential(pm, n_inp)
        H = hamiltonian(pm, v_ks=v_ks)

        # Calculate new n_out
        n_out, orbitals_out, eigenvalues_out = groundstate(pm,H)

        # Calculate the Kohn-Sham gap
        gap = eigenvalues_out[pm.sys.NE]- eigenvalues_out[pm.sys.NE-1]
        if gap < 1e-3:
            string = "\nLDA: Warning: small KS gap {:.3e} Ha. Convergence may be slow.".format(gap)
            pm.sprint(string, 1)

        # Calculate the self-consistent density and energy error
        dn = np.sum(np.abs(n_inp-n_out))*pm.space.delta
        dE = E - E_old

        # Check if converged
        converged = dn < pm.lda.tol and np.abs(dE) < pm.lda.etol
        string = 'LDA: E = {:.8f} Ha, de = {:+.3e}, dn = {:.3e}, iter = {}'.format(E, dE, dn, iteration)
        pm.sprint(string, 1, newline=False)

        # Iterate
        iteration += 1

    iteration -= 1
    pm.sprint('')

    # Print to screen
    if not converged:
        string = 'LDA: Warning: convergence not reached in {} iterations. Terminating.'.format(iteration)
        pm.sprint(string, 1)
    else:
        pm.sprint('LDA: reached convergence in {} iterations.'.format(iteration),0)

    # Self-consistent solution
    density = n_out
    orbitals = orbitals_out
    eigenvalues = eigenvalues_out
    
    # Calculate potentials and energies
    if pm.lda.NE == 'heg':
        E_xc, E_x, E_c = xc_energy(pm, density, separate=True)
        v_xc, v_x, v_c = xc_potential(pm, density, separate=True)
    else:
        E_xc = xc_energy(pm, density)
        v_xc = xc_potential(pm, density)
    v_h = hartree_potential(pm, density)
    v_ks = pm.space.v_ext + v_h + v_xc
    E = total_energy_eigf(pm, orbitals=orbitals, density=density)
    E_h = hartree_energy(pm, v_h, density)
    E_hxc = E_h + E_xc

    # Print to screen
    pm.sprint('LDA: ground-state energy: {}'.format(E),1)
    pm.sprint('LDA: ground-state Hartree exchange-correlation energy: {}'.format(E_hxc),1)
    pm.sprint('LDA: ground-state Hartree energy: {}'.format(E_h),1)
    pm.sprint('LDA: ground-state exchange-correlation energy: {}'.format(E_xc),1)
    if pm.lda.NE == 'heg':
        pm.sprint('LDA: ground-state exchange energy: {}'.format(E_x),1)
        pm.sprint('LDA: ground-state correlation energy: {}'.format(E_c),1)

    # Save the quantities to file
    results = rs.Results()
    results.add(density, 'gs_lda{}_den'.format(pm.lda.NE))
    results.add(v_h, 'gs_lda{}_vh'.format(pm.lda.NE))
    results.add(v_xc, 'gs_lda{}_vxc'.format(pm.lda.NE))
    results.add(v_ks, 'gs_lda{}_vks'.format(pm.lda.NE))
    results.add(E, 'gs_lda{}_E'.format(pm.lda.NE))
    results.add(E_xc, 'gs_lda{}_Exc'.format(pm.lda.NE))
    results.add(E_h, 'gs_lda{}_Eh'.format(pm.lda.NE))
    results.add(E_hxc, 'gs_lda{}_Ehxc'.format(pm.lda.NE))
    if pm.lda.NE == 'heg' :
        results.add(E_x, 'gs_lda{}_Ex'.format(pm.lda.NE))
        results.add(E_c, 'gs_lda{}_Ec'.format(pm.lda.NE))
        results.add(v_x, 'gs_lda{}_vx'.format(pm.lda.NE))
        results.add(v_c, 'gs_lda{}_vc'.format(pm.lda.NE))
    results.add(orbitals.T,'gs_lda{}_eigf'.format(pm.lda.NE))
    results.add(eigenvalues,'gs_lda{}_eigv'.format(pm.lda.NE))
    if pm.run.save:
        results.save(pm)

    # Propagate through real time
    if pm.run.time_dependence:

        # Construct arrays
        v_ks_td = np.zeros((pm.sys.imax,pm.space.npt), dtype=np.float)
        v_xc_td = np.zeros((pm.sys.imax,pm.space.npt), dtype=np.float)
        v_h_td = np.zeros((pm.sys.imax,pm.space.npt), dtype=np.float)
        current = np.zeros((pm.sys.imax,pm.space.npt), dtype=np.float)
        density_td = np.zeros((pm.sys.imax,pm.space.npt), dtype=np.float)
        orbitals = orbitals.astype(np.cfloat)

        # Save the ground-state
        v_ks_td[0,:] = v_ks[:]
        v_h_td[0,:] = v_h[:]
        v_xc_td[0,:] = v_xc[:]
        density_td[0,:] = density[:]

        # Perform real time iterations
        for i in range(1, pm.sys.imax):

            # Print to screen 
            string = 'LDA: evolving through real time: t = {}'.format(i*pm.sys.deltat)
            pm.sprint(string, 1, newline=False)

            # Construct the Hamiltonian
            H = hamiltonian(pm, orbitals=orbitals, perturbation=True)
            H_full = banded_to_full(pm, H)

            # Propagate through time-step using the Crank-Nicolson method 
            orbitals[:, :pm.sys.NE] = crank_nicolson_step(pm, orbitals, H_full)
            density_td[i,:] = electron_density(pm, orbitals)
            v_ks_td[i,:] = pm.space.v_ext[:] + pm.space.v_pert[:] + hartree_potential(pm, density_td[i,:]) + xc_potential(pm, density_td[i,:])

            # Hartree and exchange-correlation potential
            v_h_td[i,:] = hartree_potential(pm, density_td[i,:])
            v_xc_td[i,:] = xc_potential(pm, density_td[i,:])

        # Calculate the current density
        current_density = calculate_current_density(pm, density_td)
 
        # Save the quantities to file
        results.add(v_ks_td, 'td_lda{}_vks'.format(pm.lda.NE))
        results.add(v_h_td, 'td_lda{}_vh'.format(pm.lda.NE))
        results.add(v_xc_td, 'td_lda{}_vxc'.format(pm.lda.NE))
        results.add(density_td, 'td_lda{}_den'.format(pm.lda.NE))
        results.add(current_density, 'td_lda_cur')
        if pm.run.save:
            results.save(pm)

        pm.sprint('',1)

    return results
