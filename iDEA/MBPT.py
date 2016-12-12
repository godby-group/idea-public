"""Computes Green function and self-energy in the GW approximation

Different flavours of GW (G0W0, scGW, QSGW) are available.  The implementation
follows the GW-space-time approach detailed in [Rojas1995]_ and  [Rieger1999]_.
"""
from __future__ import division
import copy
import numpy as np
import scipy as sp
import results as rs
import mklfftwrap

class SpaceTimeGrid:
    """Stores spatial and frequency grids"""

    def __init__(self,pm):
        # (imaginary) time
        self.tau_max = pm.mbpt.tau_max
        self.tau_npt = pm.mbpt.tau_npt
        self.tau_delta = 2.0*self.tau_max / self.tau_npt

        # We use an offset grid for tau, since G0(it) is discontinuous at it=0.
        # The grid is laid out in a way appropriate for numpy's fft, and in
        # particular, this grid *always* starts with tau=dt/2.
        #   tau_grid = [dt/2,dt+dt/2,...,T/2+dt/2,-T/2+dt/2,...,-dt/2] # tau_npt odd
        #   tau_grid = [dt/2,dt+dt/2,...,T/2-dt/2,-T/2+dt/2,...,-dt/2] # tau_npt even
        #self.tau_grid = 2*self.tau_max * np.fft.fftfreq(self.tau_npt) + self.tau_delta/2.0
        self.tau_grid = 2*self.tau_max * np.fft.fftfreq(self.tau_npt)
        
        # (imaginary) frequency
        self.omega_max = np.pi / self.tau_delta
        # no shift for the frequency grid
        self.omega_grid = 2*self.omega_max * np.fft.fftfreq(self.tau_npt)
        self.omega_delta = 2*np.pi / 2*self.tau_max
        # phase factors for Fourier transform
        # numpy forward transform has minus sign in exponent
        self.phase_forward= np.exp(-1J * np.pi * np.fft.fftfreq(self.tau_npt))
        self.phase_backward= np.conj(self.phase_forward)

        # space
        self.x_max = pm.sys.xmax
        self.x_npt = pm.sys.grid
        self.x_delta = float(2*pm.sys.xmax)/float(pm.sys.grid-1)
        self.x_grid = np.linspace(-self.x_max,self.x_max,self.x_npt)

        # set up coulomb repulsion matrix v(i,j)
        tmp = np.empty((self.x_npt, self.x_npt), dtype=int)
        for i in range(self.x_npt):
            for j in range(self.x_npt):
                tmp[i,j] = np.abs(i - j)
        self.coulomb_repulsion = 1/(tmp * self.x_delta + pm.sys.acon)

        # orbitals
        self.norb = pm.mbpt.norb
        self.NE = pm.sys.NE

    def __str__(self):
        """Returns string with description of main parameters"""
        s = ""
        s += "Spatial grid: [{:.3f}, {:.3f}] in steps of dx = {:.3f}\n".\
                format(-self.x_max,self.x_max,self.x_delta)
        s += "Temporal grid: [{:.3f}, {:.3f}] in steps of dt = {:.3f}\n".\
                format(-self.tau_max,self.tau_max,self.tau_delta)
        s += "Orbitals: {} occupied, {} empty ({:.1f}% of basis set)\n".\
                format(self.NE, self.norb - self.NE, self.norb/self.x_npt*100)
        return s

        
def main(parameters):
    r"""Runs GW calculation
    
    Steps: 
       {eps_j, psi_j}
    => G0(rr';i\tau)
    => P(rr';i\tau)
    => eps(rr';i\tau)
    => eps(rr';i\omega)
    => W(rr';i\omega) (eps_inv not calculated directly)
    => W(rr';i\tau)
    => S(rr';i\tau)
    => S(rr';i\omega)
    => G(rr';i\omega) (for self-consistent calculation)
 
    """
    pm = parameters
    results = rs.Results()

    st = SpaceTimeGrid(pm)
    pm.sprint(str(st))

    # read eigenvalues and eigenfunctions of starting Hamiltonian
    h0_energies = rs.Results.read('gs_{}_eigv'.format(pm.mbpt.h0), pm)
    h0_orbitals = rs.Results.read('gs_{}_eigf'.format(pm.mbpt.h0), pm)
    h0_rho = rs.Results.read('gs_{}_den'.format(pm.mbpt.h0), pm)

    if h0_energies.dtype == np.complex:
        im_max = np.max(np.abs(h0_energies.imag))
        s  = "MBPT: Warning: single-particle energies are complex "
        s += "(maximum imaginary component: {:.3e}). ".format(im_max)
        s += "Casting to real."
        pm.sprint(s)
        h0_energies = h0_energies.real

    norb = len(h0_energies)
    if norb < st.norb:
        raise ValueError("Not enough orbitals: {} computed, {} requested.".format(norb,st.norb))
    else:
        h0_energies = h0_energies[:st.norb]
        h0_orbitals = h0_orbitals[:st.norb]

    homo = h0_energies[st.NE-1]
    lumo = h0_energies[st.NE]
    gap = lumo - homo
    pm.sprint('MBPT: single-particle gap: {:.3f} Ha'.format(gap))
    e_fermi = homo + gap / 2
    pm.sprint('MBPT: single-particle Fermi energy: {:.3f} Ha'.format(e_fermi))
    h0_energies -= e_fermi
    results.add(h0_energies, name="gs_mbpt_eigv0")
    results.add(h0_orbitals, name="gs_mbpt_eigf0")
    results.add(e_fermi, name="gs_mbpt_efermi0")

    # check that G(it) is well described
    exp_factor = np.exp(-(lumo-e_fermi)*st.tau_max)
    if exp_factor > 1e-1:
        t1 = -np.log(1e-1)/(lumo-e_fermi)
        t2 = -np.log(1e-2)/(lumo-e_fermi)
        s  = "MBPT: Warning: Width of tau-grid for G(it) is too small "
        s += "for HOMO-LUMO gap {:.3f} Ha. ".format(gap)
        s += "Increase tau_max to {:.1f} for decay to 1e-1 ".format(t1)
        s +=  "or {:.1f} for decay to 1e-2".format(t2)
        pm.sprint(s)

    # computing/reading potentials
    h0_vh = hartree_potential(st, rho=h0_rho)
    h0_vx = exchange_potential(st, orbitals=h0_orbitals)
    h0_vhxc = hartree_exchange_correlation_potential(pm.mbpt.h0, h0_orbitals, h0_vh, h0_vx, st)

    def save(O, shortname, force_dg=False):
        """Auxiliary function for saving 3d objects
        
        Note: This needs to be defined *within* main in order to avoid having
        to pass a long list of arguments
        """
        if (shortname in pm.mbpt.save_diag) or force_dg:
            name = "gs_mbpt_{}_dg".format(shortname)
            results.add(bracket_r(O, h0_orbitals, st), name)
            results.save(pm, list=[name])

        if shortname in pm.mbpt.save_full:
            name = "gs_mbpt_{}".format(shortname)
            results.add(O, name)
            results.save(pm, list=[name])

    # compute G0
    pm.sprint('MBPT: setting up G0(it)',0)
    G0, G0_pzero = non_interacting_green_function(h0_orbitals, h0_energies, st, zero='both')
    save(G0,"G0_it")

    # prepare variables
    if pm.mbpt.flavour in ['GW', 'GW0', 'QSGW']:
        # we need both G and G0 separately
        G = copy.deepcopy(G0)
        G0 = fft_t(G0, st, dir='it2if') # needed for dyson equation
        save(G0,"G0_iw")
        G_pzero = G0_pzero
        h_vh = copy.deepcopy(h0_vh)
        h_vx = copy.deepcopy(h0_vx)
        h_rho = copy.deepcopy(h0_rho)
    else:
        # we need only G0 but will call it G
        G = G0
        G_pzero = G0_pzero
        h_vh = h0_vh
        h_vx = h0_vx
        h_rho = h0_rho

        del G0, h0_vh, h0_vx, h0_rho


    ### GW self-consistency loop
    converged = False
    cycle = 0
    qp_fermi = e_fermi
    while not converged:

        pm.sprint('MBPT: setting up P(it)',0)
        P = irreducible_polarizability(G, G_pzero)
        save(P, "P{}_it".format(cycle))

        pm.sprint('MBPT: transforming P to imaginary frequency',0)
        P = fft_t(P, st, dir='it2if')
        save(P, "P{}_iw".format(cycle))


        #### testing alternative way of computing W
        # this is completely identical for flavor=dynamical
        #v_test = np.empty( (st.x_npt, st.x_npt, st.tau_npt), dtype=float)
        #for i in range(st.tau_npt):
        #    v_test[:,:,i] = st.coulomb_repulsion
        #W_test = solve_dyson_equation(v_test,P,st)
        #save(W_test, "Wt{}_iw".format(cycle))

        pm.sprint('MBPT: setting up eps(iw)',0)
        eps = dielectric_matrix(P, st)
        save(eps, "eps{}_iw".format(cycle))
        del P # not needed anymore

        pm.sprint('MBPT: setting up W(iw)',0)
        W = screened_interaction(st, epsilon=eps, w_flavour=pm.mbpt.w)
        save(W, "W{}_iw".format(cycle))
        del eps # not needed anymore

        pm.sprint('MBPT: transforming W to imaginary time',0)
        W = fft_t(W, st, dir='if2it')
        save(W, "W{}_it".format(cycle))

        pm.sprint('MBPT: computing S(it)',0)
        S = self_energy(G, W)
        save(S, "S{}_it".format(cycle))
        del W # not needed anymore

        pm.sprint('MBPT: transforming S to imaginary frequency',0)
        S = fft_t(S, st, dir='it2if')

        pm.sprint('MBPT: updating S(iw)',0)
        # real for real orbitals...
        delta = np.zeros((st.x_npt, st.x_npt), dtype=np.complex)
        np.fill_diagonal(delta, h_vh / st.x_delta)
        delta -= h0_vhxc
        if pm.mbpt.w == 'dynamical':
            # in the frequency domain we can put the exchange back
            # Note: here, we need the extrapolated G(it=0^-)
            delta += h_vx
        for i in range(st.tau_npt):
            S[:,:,i] += delta
        save(S, "S{}_iw".format(cycle), force_dg=True)

        if pm.mbpt.hedin_shift:
            pm.sprint('MBPT: performing Hedin shift',0)
            S_iw_dg = getattr(results, "gs_mbpt_S{}_iw_dg".format(cycle))
            qp_shift = 0.5 * (S_iw_dg[st.NE-1,0] + S_iw_dg[st.NE,0])
            qp_shift = qp_shift.real
            qp_fermi += qp_shift
            pm.sprint('MBPT: quasi-particle fermi energy: {:.3f} Ha ({:+.3f} Ha).'.format(qp_fermi,qp_shift))
            for i in range(st.x_npt):
                S[i,i,:] -= qp_shift / st.x_delta

        if pm.mbpt.flavour == 'G0W0':
            break
        elif cycle == pm.mbpt.max_iter:
            pm.sprint("Reached maximum number of iterations. Stopping...")
            break

        cycle = cycle + 1
        pm.sprint(''.format(cycle))
        pm.sprint('MBPT: Entering self-consistency cycle #{}'.format(cycle))

        pm.sprint('MBPT: solving the Dyson equation for new G',0)
        # note: G0 = G0(r,r';iw)
        G = solve_dyson_equation(G0, S, st)
        del S # not needed anymore

        pm.sprint('MBPT: transforming G to imaginary time',0)
        G = fft_t(G, st, dir='if2it')
        save(G, "G{}_it".format(cycle))

        # extract density
        G_mzero = G[:,:,0]
        h_rho_new = np.diagonal(G_mzero.imag)

        # compute new rho
        # check convergence...
        results.add(h_rho_new, "gs_mbpt_den{}".format(cycle))
        results.save(pm, list=["gs_mbpt_den{}".format(cycle)])

        rho_norm = np.sum(h_rho_new) * st.x_delta
        pm.sprint("MBPT: norm of new density: {:.3f} electrons".format(rho_norm))
        rho_delta_max = np.max(np.abs(h_rho_new - h_rho))
        if rho_delta_max < pm.mbpt.den_tol:
            converged = True
        else:
            pm.sprint("Max. change in rho: {:.2e} > {:.2e}".format(rho_delta_max,pm.mbpt.den_tol))
            h_rho = h_rho_new
            h_vh = hartree_potential(st, rho=h_rho)
            h_vx = -G_mzero.imag * st.coulomb_repulsion # = iGv

            # extrapolate G(it=0) from above
            eps = np.max(np.abs(G.real))
            if eps > pm.mbpt.den_tol:
                st.sprint("MBPT: Warning: Discarding real part with max. {:.3e} during extrapolation".format(eps))
            G_pzero = extrapolate_to_zero(G, st, 'from_above')
            #h_vx = h_vx
            # note: this should not be needed anymore for extrapolated h_vx
            #for i in range(st.x_npt):
            #    h_vx[i,i] -= qp_shift / st.x_delta

        #break
    


    #if pm.mbpt.flavour == 'G0W0':

    #    results = G0W0(pm, results)
    #elif pm.mbpt.flavour == 'GW':
    #    results = SCGW(pm, results)
    #elif pm.mbpt.flavour == 'QSGW':
    #    results = QSGW_imaginary_time(orbitals, sp_energies, pm, results)
    #else:
    #    raise ValueError("MBPT mode {} is not implemented".format(pm.mbpt.flavour))

    return results

def hartree_exchange_correlation_potential(h0, orbitals, h0_vh, h0_vx, st):
    r"""Returns Hartree-exchange-correlation potential of h0

    .. math ::
        
        \mathcal{H}_0 = T + V_{ext}(r) + V_{Hxc}(r,r') \\
        V_{Hxc}(r,r') = \delta(r-r')V_H(r) + V_x(r,r') + V_c(r,r')

    Possible choices for h0 are
      * 'LDA'/'EXT': :math:`V_{Hxc}(r,r') = \delta(r-r') (V_H(r) + V_{xc}(r))`
      * 'H': :math:`V_{Hxc}(r,r') = \delta(r-r') V_H(r)`
      * 'HF': :math:`V_{Hxc}(r,r') = \delta(r-r') V_H(r) + V_x(r,r')`
      * 'NON': :math:`V_{Hxc}(r,r') = 0

    parameters
    ----------
    h0: string
      The choice of single-particle Hamiltonian h0
    h0: string
        input parameters
    st: object
        space-time grid
    orbitals : array_like
        orbitals of non-interacting hamiltonian
    h0_vh: array_like
        Hartree potential V_H(r) of non-interacting density
    h0_vx: array_like
        Fock exchange operator V_x(r,r') of non-interacting density
    """

    h0_vhxc = np.zeros((st.x_npt, st.x_npt), dtype=np.float)
    if h0 == 'non':
        # non-interacting: v_Hxc = 0
        np.fill_diagonal(h0_vhxc, np.zeros(st.x_npt))
    elif h0 == 'h':
        # Hartree: v_Hxc = v_H
        tmp = rs.Results.read('gs_{}_vh'.format(h0), pm)
        np.fill_diagonal(h0_vhxc, tmp / st.x_delta)
    elif h0 == 'lda' or h0 == 'ext':
        # KS-DFT: v_Hxc = v_H + v_xc
        tmp = rs.Results.read('gs_{}_vh'.format(h0), pm)
        tmp += rs.Results.read('gs_{}_vxc'.format(h0), pm)
        np.fill_diagonal(h0_vhxc, tmp / st.x_delta)
    elif h0 == 'hf':
        # Hartree-Fock: v_Hxc = v_H + v_x
        np.fill_diagonal(h0_vhxc, h0_vh / st.x_delta)
        h0_vhxc += h0_vx
    else:
        raise ValueError("Unknown h0 flavor '{}'".format(h0))

    return h0_vhxc

def hartree_potential(st, rho=None, G=None):
    r"""Sets up Hartree potential V_H(r) from electron density.

    .. math::

       V_H(r) = \int \frac{\rho(r')}{|r-r'|} dr' = (-i) \int \frac{G(r',r';0)}{|r-r'|}dr'

    Note: :math:`V_H(r,r';i\tau) = \delta(r-r')\delta(i\tau)V_H(r)` with
    :math:`\delta(i\tau)=i\delta(\tau)`.
    """
    if rho is not None:
        pass
    elif G is not None:
        #TODO: to implement!
        rho = electron_density(pm, G=G)
    else:
        raise IOError("Need to provide either rho or G.")

    v_h = np.dot(st.coulomb_repulsion, rho) * st.x_delta
    return v_h

def exchange_potential(st, G=None, orbitals=None):
    r"""Calculate Fock exchange operator V_x(r,r')

    Can take either the Green function G(it) as input or the orbitals.

    parameters
    ----------
    st: object
       space-time grid
    G: array_like
       Green function G(r,r';it) (or G(r,r';t))
    orbitals: array_like
       single-particle orbitals

    returns v_x(r,r')
    """
    if G is not None:
        #TODO: need extrapolated G here!
        # default multiplication is element-wise
        v_x = 1J * G[:, :, 0] * st.coulomb_repulsion
    elif orbitals is not None:
        v_x = np.zeros((st.x_npt,st.x_npt),dtype=complex)
        for i in range(st.NE):
            orb = orbitals[i]
            v_x -= np.tensordot(orb.conj(), orb, axes=0)
        v_x = v_x * st.coulomb_repulsion
    else:
        raise IOError("Need to provide either G or orbitals.")

    return v_x


def non_interacting_green_function(orbitals, energies, st, zero='0-'):
    r"""Calculates non-interacting Green function G0(r,r';it).

    :math:`G_0(r,r';i\tau)` is constructed from a set of eigenvectors
    and eigenenergies of a single-particle Hamiltonian in imaginary time.

    .. math ::
        G_0(r,r';i\tau) = (-i) \sum_s^{empty} \varphi_s(r) \varphi_s(r') e^{-\varepsilon_s\tau} \theta(\tau)
                        +   i  \sum_s^{occupied} \varphi_s(r) \varphi_s(r') e^{-\varepsilon_s\tau} \theta(-\tau)


    See equation 3.3 of [Rieger1999]_. Note that we have reversed the sign of
    :math:`\tau` in order to be consistent with Hedin [Hedin1970]_.

    FLOPS: norb * (grid**2 + 2 * tau_npt * grid**2)

    parameters
    ----------
    orbitals : array
        set of single-particle orbitals
    energies : array
        corresponding single-particle energies
    st : object
        contains space-time parameters
    zero : string
        How to treat it=0
        '0+': :math:`G(0) = \lim_{t\downarrow 0}G(it)`,
            determined by empty states
        '0-': :math:`G(0) = \lim_{t\uparrow 0}G(it)`,
            determined by occupied states with :math:`(-i)G(r,r,0)=\rho(r)`
        'both': return '0-' Green function *and* it=0 slice of '0+' Green function
    """
    coef = np.zeros((st.norb,st.tau_npt), dtype=complex)
    coef_zero = np.zeros(st.norb, dtype=complex)
    for i in range(st.norb):
        en = energies[i]

        # first handle special case tau=0 (k=0)
        if zero == '0+' and en > 0:
            # put empty states into tau=0
            coef[i,0] = -1J
        elif zero == '0-' and en < 0:
            # put occupied states into tau=0
            coef[i,0] = +1J
        elif zero == 'both':
            if en > 0:
                # put empty states into tau=0 slice
                coef_zero[i] = -1J
            elif en < 0:
                # put occupied states into tau=0
                coef[i,0] = +1J

        # note: this could still be vectorized
        for k in range(1,st.tau_npt):
            tau = st.tau_grid[k]

            if en > 0 and tau > 0:
                # tau > 0, empty states
                coef[i,k] = -1J * np.exp(-en * tau)
            elif en < 0 and tau <= 0:
                # tau < 0, occupied states
                coef[i,k] = +1J * np.exp(-en * tau)

    # one call to np.dot with reshape
    orb_mat = np.empty( (st.x_npt, st.x_npt, st.norb), dtype=complex )
    for i in range(st.norb):
        orb = orbitals[i]
        orb_mat[:,:,i] = np.tensordot(orb.conj(), orb,axes=0)
    # for some reason, einsum is significantly slower...
    orb_mat_r = orb_mat.reshape(st.x_npt*st.x_npt, st.norb)
    #G0 = np.einsum('ij,jl->il',orb_mat.reshape(st.x_npt*st.x_npt, norb),coef)
    G0 = np.dot(orb_mat_r,coef).reshape(st.x_npt,st.x_npt,st.tau_npt)

    if zero == 'both':
        G0_pzero = np.dot(orb_mat_r,coef_zero).reshape(st.x_npt, st.x_npt)
        return G0, G0_pzero

    else:
        return G0


def bracket_r(O, orbitals, st, mode='diagonal'):
    r"""Calculate expectationvalues of O(r,r';t) for each t wrt orbitals

    .. math:: O_{jj}(t) = \langle \varphi_j | O(r,r';t) | \varphi_j\rangle
                        = \int \varphi_j^*(r) O(r,r';t)\varphi_j(r')\,dr\,dr'

    Note: For non-hermitian O, the pairing of r,r' with \varphi_j^*,\varphi_j
    matters.

    parameters
    ----------
    O: array
        Operator O(r,r';t)
    orbitals: array
        Array of shape (norb, grid) containing orbtials
    pm: object
        Parameters object
    mode: string
        if 'diagonal', evaluates <j|O|j> for all j
        if 'matrix', evaluates <i|O|j> for all i,j

    Returns
        bracket_r[i,j]:  i=orbital index, j=index of temporal grid
    """
    orbs = copy.copy(orbitals) * st.x_delta  # factor needed for integration

    # Performing one matrix-matrix multiplication is substantially faster than
    # performing t matrix-vector multiplications.
    # Note: If we do not np.reshape, np.dot internally performs matrix-vector
    # multiplications here.

    # numpy.dot(a,b) sums over last axis of a and 2nd-to-last axis of b
    # tmp.shape == (norb, x_npt*tau_npt)
    tmp = np.dot(orbs, O.reshape((st.x_npt, st.x_npt * st.tau_npt)))

    if mode == 'diagonal':
        # Then, we do element-wise multiplication + summation over the grid axis
        # to get the scalar product.
        bracket_r = (orbs.conj()[:,:,None] * tmp.reshape((st.norb,st.x_npt,st.tau_npt))).sum(1)
    elif mode == 'matrix':
        # bracket_r.shape == (norb,norb,t)
        bracket_r  = np.dot(orbs.conj(), tmp.reshape((st.norb,st.x_npt,st.tau_npt)))
    else:
        raise ValueError("Unknown mode {}".format(mode))
        
    return bracket_r


def fft_t(F, st, dir, phase_shift=False):
    r"""Performs 1d Fourier transform of F(r,r';t) along time dimension.

    Can handle forward & backward transforms in real & imaginary time.
    Here, we replicate the convention of [Rieger1999]_ (see equations 3.1 and 3.2)

    .. math::

        \begin{align}
        F(\omega) &= \int dt F(t) e^{i\omega t} \\
        F(t) &= \int \frac{d\omega}{2\pi} F(\omega) e^{-i\omega t}\\
        F(i\omega) &= -i\int dt F(it) e^{-i\omega t}\\
        F(it) &= i\int \frac{d\omega}{2\pi} F(i\omega) e^{i\omega t}
        \end{align}

    The infinitesimals :math:`d\tau,d\omega/2\pi` are automatically
    included in the Fourier transforms.

    Note: We adopt the Fourier transform convention by Rieger, Hedin et al.,
    which uses negative imaginary exponents for the *backward* transform in real time.
    This differs from the more common convention (adopted by numpy) of using
    negative exponents for the *forward* transform.

    numpy by default scales the forward transform by 1/n. See also
    http://docs.scipy.org/doc/numpy/reference/routines.fft.html#implementation-details.
    The MKL scales neither forward nor backward transform.

    FLOPS: tau_npt * grid**2 * (log(grid) + 2)

    parameters
    ----------
      F : array
        will be transformed along last axis
      dir : string
        't2f' time to frequency domain
        'f2t' frequency to time domain
        'it2if' imaginary time to imaginary frequency domain
        'if2it' imaginary frequency to imaginary time domain
      phase_shift: bool
        True: use with shifted tau grid (tau_grid[0] = tau_delta/2)
        False: use with unshifted tau grid (tau_grid[0] = 0)
    """

    n = float(F.shape[-1])

    # Crazy - taking out the prefactors p really makes it faster    
    if dir == 't2f':
        out = mklfftwrap.ifft_t(F) * st.tau_delta
        #out = np.fft.ifft(F, axis=-1) * n * st.tau_delta
        if phase_shift:
            out *= st.phase_backward
    elif dir == 'f2t':
        p = 1 / (n * st.tau_delta)
        if phase_shift:
            out = mklfftwrap.fft_t(F * st.phase_forward) * p
        else:
            out = mklfftwrap.fft_t(F) * p
            #out = np.fft.fft(F, axis=-1) / (n * st.tau_delta)
    elif dir == 'it2if':
        p = -1J * st.tau_delta
        out = mklfftwrap.fft_t(F) * p
        #out = -1J * np.fft.fft(F, axis=-1) * st.tau_delta
        if phase_shift:
            out *= st.phase_forward
    elif dir == 'if2it':
        p = 1J / (n * st.tau_delta)
        if phase_shift:
            out = mklfftwrap.ifft_t(F * st.phase_backward) * p
        else:
            out = mklfftwrap.ifft_t(F) * p
            #out = 1J * np.fft.ifft(F, axis=-1) / st.tau_delta
    else:
        raise IOError("FFT direction {} not recognized.".format(dir))

    return out


def irreducible_polarizability(G, G_pzero):
    r"""Calculates irreducible polarizability P(r,r',it).

    .. math:: P(r,r';i\tau) = -iG(r,r';i\tau) G(r',r;-i\tau)

    parameters
    ----------
    G : array
        Green function
    G_pzero : array
        it=0 component of Green function with :math:`G(0) = \lim_{t\downarrow 0}G(it)`

    FLOPS: grid**2 * tau_npt * 3

    See equation 3.4 of [Rieger1999]_.
    """

    G_rev = copy.copy(G)
    G_rev = G_rev.swapaxes(0,1)
    G_rev[:,:,0] = G_pzero
    # need t=0 to become the *last* index for ::-1
    G_rev = np.roll(G, -1, axis=2)
    P =  -1J * G * G_rev[:,:,::-1]

    # v2 done in python, significantly slower...
    #P = np.zeros((pm.grid, pm.grid, pm.tau_npt), dtype=complex)
    #for i in range(pm.grid):
    #    for j in range(pm.grid):
    #        for k in range(pm.tau_npt):
    #            P[i, j, k] = -1J * G_m[i, j, k] * G_m[j, i, -k]

    return P


def dielectric_matrix(P, st):
    r"""Calculates dielectric matrix eps(r,r';iw) from polarizability.

    .. math::

        \varepsilon(r,r';i\omega) = \delta(r-r') - \int v(r-r'')P(r'',r';i\omega)d^3r''

    See equation 3.5 and 4.3 of [Rieger1999]_.

    FLOPS: tau_npt * (2*grid * grid**2)

    parameters
    ----------
    P: array_like
      irreducible polarizability P(r,r';iw)
    st: object
      space-time grid
    """
    v = st.coulomb_repulsion

    # Note: dot + reshape is a tiny bit faster than tensordot...
    #eps = - np.tensordot(v, P, axes=(1,1)) * st.x_delta
    eps = - np.dot(v, P.reshape(st.x_npt, st.x_npt*st.tau_npt)) * st.x_delta
    eps = eps.reshape(st.x_npt, st.x_npt, st.tau_npt)

    # add delta(r-r')
    tmp = 1 / st.x_delta
    for i in range(st.x_npt):
        eps[i, i, :] += tmp

    # v2 within python
    #for k in range(st.tau_npt):
    #    eps[:, :, k] = -np.dot(v, P[:, :, k]) * st.deltax

    #    # add delta(r-r')
    #    for i in range(st.x_npt):
    #        eps[i, i, k] += 1 / st.deltax

    return eps


def screened_interaction(st, epsilon_inv=None, epsilon=None, w_flavour='full'):
    r"""Calculates screened interaction W(r,r';iw).

    Input is the (inverse) dielectric function.

    .. math::

        W(r,r';i\omega) = \int v(r-r'') \varepsilon^{-1}(r'',r';i\omega)d^3r''

    See equation 3.6 of [Rieger1999]_.

    FLOPS: tau_npt * (grid**3 + grid**2)

    parameters
    ----------
    epsilon_inv: array_like
        inverse dielectric matrix eps_inv(r,r',iw)
        if provided, we compute W = epsilon_inv v
    epsilon: array_like
        dielectric matrix eps(r,r',iw)
        if provided, we solve epsilon W = v instead
    w_flavour: string
        'full': for full screened interaction (static and dynamical parts)
        'dynamical': dynamical part only: W = (eps_inv -1) v

    returns W
    """
    W = np.empty((st.x_npt, st.x_npt, st.tau_npt), dtype=complex)
    v = st.coulomb_repulsion

    if w_flavour not in ['full', 'dynamical']:
        raise ValueError("Unrecognized flavour {} for screened interaction".format(w_flavour))

    if epsilon_inv is not None:
        # W = eps_inv * v

        if w_flavour == 'dynamical':
            # calculate only dynamical part of S: W = (eps_inv-1) v
            tmp = 1.0 / st.x_delta
            for i in range(st.x_npt):
                epsilon_inv[i,i,:] -= tmp

        for k in range(st.tau_npt):
            W[:, :, k] = np.dot(epsilon_inv[:, :, k], v) * st.x_delta
        # for some strange reason, above loop is slightly *faster* than np.tensordot
        #W = np.tensordot(v, epsilon_inv, axes=(0,1)) * st.x_delta

    elif epsilon is not None:
        # solve linear system
        v_dx = v / st.x_delta

        if w_flavour == 'dynamical':
            # solve eps * (W+v) = v/dx
            for k in range(st.tau_npt):
                W[:, :, k] = np.linalg.solve(epsilon[:,:,k], v_dx) - v
        else:
            # solve eps*W = v/dx

            for k in range(st.tau_npt):
                W[:, :, k] = np.linalg.solve(epsilon[:,:,k], v_dx)

    else:
        raise ValueError("Need to provide either epsilon or epsilon_inv")

    return W


def self_energy(G, W):
    r"""Calculate the self-energy S(it) within the GW approximation.

    .. math::

        \Sigma(r,r';i\tau) = iG(r,r';i\tau)W(r,r';i\tau)

    FLOPS: tau_npt * (2*grid**2)

    See equation 3.7 of [Rieger1999]_.

    parameters
    ----------
    G: array_like
       Green function G(it)
    W: array_like
       Screened interaction W(it)

    return S
    """
    S = 1J * G * W
    return S



def solve_dyson_equation(G0, S, st):
    r"""Solves the Dyson equation for G

    .. math::

        G(r,r';i\omega) = \int  \left(\delta(r-r'') - \int  G_0(r,r''';i\omega)
          \Sigma(r''',r'';i\omega) d^3r''' \right)^{-1} G_0(r'',r';i\omega)

    parameters
    ----------
    G0: array_like
      non-interacting Green function G0(r,r';iw)
    S: array_like
      many-body self-energy S(r,r';iw)
    st: object
      space-time grid parameters

    returns updated Green function G(r,r';iw)
    """
    # 1. Compute A = (1/dx - G0*S*dx) * dx
    # note: A could be made just np.empty((st.x_npt,st.x_npt)),
    # but this would mean we can't use inverse_r
    A = np.empty((st.x_npt,st.x_npt,st.tau_npt), dtype=complex)
    pref = st.x_delta**2
    for k in range(st.tau_npt):
        A[:,:,k] = -np.dot(G0[:,:,k], S[:,:,k]) * pref

        # Note: the following einsum is equivalent but much slower
        # (probably einsum first computes scalar products between different k
        # although it returns only those from the same k in the end)
        #A = -np.einsum('ijk,jlk->ilk',G0, S) * pref

    for i in range(st.x_npt):
        A[i,i,:] += 1.0



    # 2. Solve G = (A/dx)**(-1) / dx**2 * G0 * dx <=> A*G = G0
    G = np.empty((st.x_npt,st.x_npt,st.tau_npt), dtype=complex)
    for k in range(st.tau_npt):
        G[:,:,k] = np.linalg.solve(A[:,:,k], G0[:,:,k])

    ## same as above but using matrix inversion
    #A = inverse_r(A, st)
    #for k in range(st.tau_npt):
    #    G[:,:,k] = np.dot(A[:,:,k], G0[:,:,k]) * st.x_delta

    return G

def extrapolate_to_zero(F, st, dir='from_below', order=3, points=3):
    """Extrapolate quantities to time point zero

    parameters
    ----------
    F: array_like
      quantity to extrapolate
    dir: string
      'from_below': extrapolate from negative times (default)
      'from_above': extrapolate from positive times (default)
    order: int
      order of polynomial fit (order+1 parameters)
    points: int
      number of data points for polynomial fit
   
    returns extrapolated value at time zero
    """
    if dir == 'from_below':
        istart = st.tau_npt - order - 1
        iend = st.tau_npt
    elif dir == 'from_above':
        istart = 1
        iend = 1 + order + 1

    #TODO: optimise for performance
    out = np.zeros((st.x_npt,st.x_npt), dtype=np.float)
    for i in xrange(0,st.x_npt):
        for j in xrange(0,st.x_npt):
           x = st.tau_grid[istart:iend]
           y = F[i,j, istart:iend].imag
           z = np.poly1d(np.polyfit(x, y, order))  
           out[i,j] = z(0)
    return 1J * out


#def adjust_self_energy(S, v_h, h0_vxc, st, w_flavour='dynamical'):
#    r"""Calculate the self-energy S(it) within the GW approximation.
#
#    Here, the term "self-energy" is meant in the many-body sense, i.e.
#    :math:`\Delta\Sigma = \mathcal{H}-\mathcal{H}^0` where `\mathcal{H}` is the
#    many-body Hamiltonian and `\mathcal{H}^0` the single-particle Hamiltonian
#    (of whatever kind).
#    This is the self-energy required by the Dyson equation.
#
#    In general, we have
#
#    .. math::
#
#        \Sigma(r,r';i\tau) = iG(r,r';i\tau)W(r,r';i\tau) \\
#        \Delta\Sigma(r,r';i\omega) = \Sigma(r,r';i\omega) + \delta(r-r') V_H(r) - V_{Hxc}^0(r,r')
#
#    where the superscript 0 indicates that the corresponding operator is 
#    evaluated for the non-interacting Hamiltonian.
#
#    Note: :math:`V_H(r,r';i\tau) = \delta(r-r')\delta(i\tau)V_H(r)` with
#    :math:`\delta(i\tau)=i\delta(\tau)`.
#
#
#    parameters
#    ----------
#    S: array_like
#       S(iw) where S(it) = -iGW
#    vh: array_like
#       up-to-date Hartree potential
#    h0: string
#      The choice of single-particle Hamiltonian h0
#       'H'  :  h0 contains external & Hartree potential  (default)
#       'HF' :  h0 contains external, Hartree & exchange potential
#       'KS' :  h0 contains external, Hartree & exchange-correlation potential
#       'NON':  h0 contains only external potential
#    w_flavor: string
#      flavor of screened interaction
#       'full':  including the static exchange interaction
#       'dynamical':  not including the static exchange interaction
#    self_consistent:  True, if performing self-consistent calculation
#    rho0:  single-particle electron-density
#        needed for self-consistent calculations
#    Vxc0:  single-particle exchange-correlation potential
#        needed for self-consistent calculations
#
#    """
#    # constructing static adjustment of self-energy
#    diff = np.zeros((st.x_npt, st.x_npt), dtype=np.float)
#
#    np.fill_diagonal(diff, v_h)
#    diff -= h0_vhxc
#
#    if w_flavour == 'dynamical':
#        # in the frequency domain, we can put the exchange back
#        # Note: here, we need the extrapolated G(it=0^-)
#        diff += exchange_potential(st,G=G)
#
#    for i in range(st.tau_npt):
#        S[:,:,i] += diff
#
#    return S


#
## Function to calculate coulomb interaction in the frequency domain
#def coulomb_interaction(st):
#   v_f = np.zeros((st.tau_N,st.x_N,st.x_N), dtype='complex')
#   for i in xrange(0,st.x_N):
#      for j in xrange(0,st.x_N):
#         v_f[:,i,j] = 1.0/(abs(st.x_grid[j]-st.x_grid[i])+pm.sys.acon) # Softened coulomb interaction
#   return v_f
#
## Function to calculate the irreducible polarizability P in the time domain
#def irreducible_polarizability(st,G,iteration):
#   P = np.zeros((st.tau_N,st.x_N,st.x_N), dtype='complex')
#   for k in xrange(0,st.tau_N):
#      for i in xrange(0,st.x_N):
#         for j in xrange(0,st.x_N):
#            P[k,i,j] = -1.0j*G[k,i,j]*G[-k-1,j,i]
#   return P
#
## Function to calculate the screened interaction W_f in the frequency domain
#def screened_interaction(st,v_f,P_f,iteration):
#   W_f = np.zeros((st.tau_N,st.x_N,st.x_N), dtype='complex')
#   for k in xrange(0,st.tau_N): # At each frequency slice perform W = (I - v P)^-1 v
#      W_f[k,:,:] = np.dot(npl.inv(np.eye(st.x_N,dtype='complex') - np.dot(v_f[k,:,:],P_f[k,:,:])*st.dx*st.dx),v_f[k,:,:])
#   return W_f
#
## Function to calculate the self energy S in the time domain
#def self_energy(st,G,W,iteration):
#   S = np.zeros((st.tau_N,st.x_N,st.x_N), dtype='complex')
#   S[:,:,:] = 1.0j*G[:,:,:]*W[:,:,:] # GW approximation
#   return S
#
## Function to perfrom the hedin shift in the frequency domain
#def hedin_shift(st,S_f,occupied,empty):
#   state = np.zeros(st.x_N, dtype='complex')
#   state[:] = occupied[:,-1]
#   expectation_value1 = np.vdot(state,np.dot(S_f[0,:,:],state.transpose()))*st.dx*st.dx # Calculates the expectation value of S(0) in the HOMO state
#   state[:] = empty[:,0]
#   expectation_value2 = np.vdot(state,np.dot(S_f[0,:,:],state.transpose()))*st.dx*st.dx # Calculates the expectation value of S(0) in the LUMO state
#   expectation_value = (expectation_value1 + expectation_value2)/2.0 # Calculates the expectation value of S(0) at the fermi energy
#   for i in xrange(0,st.x_N):
#      S_f[:,i,i] = S_f[:,i,i] - expectation_value/st.dx
#   return S_f
#
## Function to solve the dyson equation in the frequency domain to update G
#def dyson_equation(st,G0_f,S_f,iteration):
#   G_f = np.zeros((st.tau_N,st.x_N,st.x_N), dtype='complex') # Greens function in the frequency domain
#   for k in xrange(0,st.tau_N): # At each frequency slice perform G = (I - G0 S)^-1 G0
#      G_f[k,:,:] = np.dot(npl.inv(np.eye(st.x_N,dtype='complex') - np.dot(G0_f[k,:,:],S_f[k,:,:])*st.dx*st.dx),G0_f[k,:,:])
#   return G_f
#
#
#
## Main function
#def main(parameters):
#   global pm
#   pm = parameters
#
#   # Construct space-time grid
#   st = SpaceTime()
#   
#   # Construct the kinetic energy
#   K = constructK(st)
#
#   # Construct the potential
#   V = constructV(st)
#
#   # Constuct the hamiltonian
#   H = K + V 
#
#   # Compute all wavefunctions
#   print 'MBPT: computing eigenstates of single particle hamiltonian'
#   solution = spsla.eigsh(H, k=pm.sys.grid-3, which='SA', maxiter=1000000)
#   energies = solution[0] 
#   wavefunctions = solution[1]
#
#   # Normalise all wavefunctions
#   length = len(wavefunctions[0,:])
#   for i in xrange(0,length):
#      wavefunctions[:,i] = wavefunctions[:,i]/(np.linalg.norm(wavefunctions[:,i])*pm.sys.deltax**0.5)
#
#   # Make array of occupied wavefunctions
#   occupied = np.zeros((pm.sys.grid,pm.sys.NE), dtype='complex')
#   occupied_energies = np.zeros(pm.sys.NE)
#   for i in xrange(0,pm.sys.NE):
#      occupied[:,i] = wavefunctions[:,i]
#      occupied_energies[i] = energies[i]
#
#   # Make array of empty wavefunctions
#   empty = np.zeros((pm.sys.grid,pm.mbpt.number_empty), dtype='complex')
#   empty_energies = np.zeros(pm.mbpt.number_empty)
#   for i in xrange(0,pm.mbpt.number_empty):
#      s = i + pm.sys.NE
#      empty[:,i] = wavefunctions[:,s]
#      empty_energies[i] = energies[s]
#
#   # Re-scale energies
#   E1 = occupied_energies[-1] 
#   E2 = empty_energies[0]
#   E = (E2+E1)/2.0
#   occupied_energies[:] = occupied_energies[:] - E
#   empty_energies[:] = empty_energies[:] - E
#
#   # Compute phase factors for this grid
#   phase_factors = generate_phase_factors(st)
#
#   # Calculate the coulomb interaction in the frequency domain
#   print 'MBPT: computing coulomb interaction v'
#   v_f = coulomb_interaction(st) 
#
#   # Calculate the non-interacting green's function in imaginary time
#   G0 = non_interacting_greens_function(st, occupied, occupied_energies, empty, empty_energies)
#   G0_f = fourier(st,G0,0,phase_factors) 
#
#   # Initial guess for the green's function
#   print 'MBPT: constructing initial greens function G'
#   G = np.zeros((st.tau_N,st.x_N,st.x_N), dtype='complex')
#   G[:,:,:] = G0[:,:,:]
#
#   # Determine level of self-consistency
#   converged = False
#   iteration = 0
#   if(pm.mbpt.self_consistent == 0):
#      max_iterations = 1
#   if(pm.mbpt.self_consistent == 1):
#      max_iterations = pm.mbpt.max_iterations
#
#   # GW self-consistency loop
#   print 'MBPT: performing first iteration (one-shot)'
#   while(iteration < max_iterations and converged == False):
#      if(iteration == 0 or pm.mbpt.update_w == True):
#         P = irreducible_polarizability(st,G,iteration) # Calculate P in the time domain
#         P_f = fourier(st,P,0,phase_factors) # Fourier transform to get P in the frequency domain
#         W_f = screened_interaction(st,v_f,P_f,iteration) # Calculate W in the frequency domain
#         W = fourier(st,W_f,1,phase_factors) # Fourier transform to get W in the time domain
#      S = self_energy(st,G,W,iteration) # Calculate S in the time domain
#      S_f = fourier(st,S,0,phase_factors) # Fourier transform to get S in the frequency domain
#      S_f = correct_diagrams(st,S_f,v_f,extract_density(st,G)) # Correct diagrams to S in the frequency domain
#      S_f = hedin_shift(st,S_f,occupied,empty) # Apply the hedin shift to S in the frequency domain
#      G_f = dyson_equation(st,G0_f,S_f,iteration) # Solve the dyson equation in the frequency domain
#      G = fourier(st,G_f,1,phase_factors) # Fourier transform to get G in the time domain
#      if(iteration > 0):
#         converged = has_converged(density,extract_density(st,G),iteration) # Test for converence
#      density = extract_density(st,G) # Extract the ground-state density from G
#      iteration += 1
#
#   # Extract the ground-state density from G
#   if(pm.mbpt.self_consistent == 1):
#      print
#   print 'MBPT: computing density from the greens function G'
#   density = extract_density(st,G)
#
#   # Normalise the density
#   print 'MBPT: normalising density by ' + str(float(pm.sys.NE)/(np.sum(density)*st.dx))
#   density[:] = (density[:]*float(pm.sys.NE))/(np.sum(density)*st.dx)
#
#   # Output ground state density
#   results = rs.Results()
#   results.add(density,'gs_mbpt_den')
#   if pm.run.save:
#      results.save(pm)
#      
#   # Output all hedin quantities
#   if(pm.mbpt.output_hedin == True):
#      output_quantities(G0,G0_f,P,P_f,W_f-v_f,S_f,G) 
#   
#   return results
