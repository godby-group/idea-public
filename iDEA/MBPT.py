"""Computes ground-state charge density in the GW approximation

Different flavours of the GW approximation of many-body perturbation theory
(G0W0, GW, GW0) are available.  The implementation follows the GW-space-time
approach detailed in [Rojas1995]_ and  [Rieger1999]_.

Besides the ground-state charge density, the code also computes quasiparticle
energies and, if desired, the Green function of the system.

"""


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import copy
import numpy as np
import scipy as sp
import scipy.linalg as spla
import scipy.sparse as sps
from . import results as rs
from . import continuation
from .fftwrap import fft_1d, ifft_1d
import iDEA.HF


class SpaceTimeGrid(object):
    """Stores spatial and frequency grids"""

    def __init__(self,pm):

        # (imaginary) time
        self.tau_max = pm.mbpt.tau_max
        self.tau_npt = pm.mbpt.tau_npt
        self.tau_delta = 2.0*self.tau_max / self.tau_npt
        self.tau_grid = 2*self.tau_max * np.fft.fftfreq(self.tau_npt)

        # For offset grid (no longer used): offset grid for tau, since G0(it) is discontinuous at it=0.
        # The grid is laid out in a way appropriate for numpy's fft, and in
        # particular, this grid *always* starts with tau=dt/2.
        #   tau_grid = [dt/2,dt+dt/2,...,T/2+dt/2,-T/2+dt/2,...,-dt/2] # tau_npt odd
        #   tau_grid = [dt/2,dt+dt/2,...,T/2-dt/2,-T/2+dt/2,...,-dt/2] # tau_npt even
        #self.tau_grid = 2*self.tau_max * np.fft.fftfreq(self.tau_npt) + self.tau_delta/2.0

        # (imaginary) frequency
        self.omega_max = np.pi / self.tau_delta
        self.omega_npt= self.tau_npt
        self.omega_delta = (2*np.pi) / (2*self.tau_max)
        self.omega_grid = 2*self.omega_max * np.fft.fftfreq(self.tau_npt)

        # phase factors for Fourier transform (no longer used)
        # numpy forward transform has minus sign in exponent
        #self.phase_forward= np.exp(-1J * np.pi * np.fft.fftfreq(self.tau_npt))
        #self.phase_backward= np.conj(self.phase_forward)

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
        self.coulomb_repulsion = 1.0/(tmp * self.x_delta + pm.sys.acon)

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


class Container(object):
    """Stores quantities for GW cycle"""
    pass


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
    => G(rr';i\omega)

   parameters
   ----------
   parameters : object
      Parameters object

   returns object
      Results object
    """
    pm = parameters
    if pm.mbpt.flavour in ['GW','G0W0','GW0','QSGW']:
        pm.sprint('MBPT: running {} calculation'.format(pm.mbpt.flavour),1)
    else:
        raise ValueError("Unknown MBPT flavour {}".format(pm.mbpt.flavour))
    results = rs.Results()
    st = SpaceTimeGrid(pm)
    pm.sprint(str(st),0)

    # read eigenvalues and eigenfunctions and potentials of starting Hamiltonian
    pm.sprint('MBPT: starting from {} approximation'.format(pm.mbpt.h0),1)
    h0 = read_input_quantities(pm,st)
    results.add(h0.energies, name="gs_mbpt_eigv0")
    results.add(h0.orbitals, name="gs_mbpt_eigf0")
    results.add(h0.e_fermi, name="gs_mbpt_efermi0")

    # Note: This needs to be defined *within* main in order to avoid having
    # to pass a long list of arguments
    def save(O, shortname, force_dg=False):
        """Auxiliary function for saving 3d objects

        """
        if (shortname in pm.mbpt.save_diag) or force_dg:
            name = "gs_mbpt_{}_dg".format(shortname)
            results.add(bracket_r(O, h0.orbitals, st), name)
            if pm.run.save:
                results.save(pm, list=[name])

        if shortname in pm.mbpt.save_full:
            name = "gs_mbpt_{}".format(shortname)
            results.add(O, name)
            if pm.run.save:
                results.save(pm, list=[name])

        if shortname in pm.mbpt.save_zero:
            try:
                name = "gs_mbpt_{}_0".format(shortname)
                results.add(O[:,:,0], name)
            except IndexError:
                name = "gs_mbpt_{}_0".format(shortname)
                results.add(O, name)
            if pm.run.save:
                results.save(pm, list=[name])

    # compute G0
    pm.sprint('MBPT: setting up G0(it)',0)
    G0, G0_pzero = non_interacting_green_function(h0.orbitals, h0.energies, st, zero='both')
    save(G0,"G0_it")

    # prepare variables
    G = copy.deepcopy(G0)
    G0 = fft_t(G0, st, dir='it2if') # needed for dyson equation
    save(G0,"G0_iw")
    G_pzero = G0_pzero
    H = copy.deepcopy(h0)

    # GW self-consistency loop
    cycle = 0

    while True:

        # For GW0, no need to recompute W
        if not (pm.mbpt.flavour == 'GW0' and cycle > 0):

            # compute P
            pm.sprint('MBPT: setting up P(it)',0)
            P = irreducible_polarizability(st, G, G_pzero, screening=pm.mbpt.screening)
            save(P, "P{}_it".format(cycle))
            save(P, "P_it")

            pm.sprint('MBPT: transforming P to imaginary frequency',0)
            if pm.mbpt.screening == 'static':
                # In static screening we still need P along all iw.
                P = fft_t(P, st, dir='it2if', screening='dynamic')
            else:
                P = fft_t(P, st, dir='it2if', screening=pm.mbpt.screening)
            save(P, "P{}_iw".format(cycle))
            save(P, "P_iw")

            # compute eps
            pm.sprint('MBPT: setting up eps(iw)',0)
            eps = dielectric_matrix(P, pm.mbpt.screening, st)
            save(eps, "eps{}_iw".format(cycle))
            save(eps, "eps_iw")
            del P # not needed anymore

            # compute W-v
            pm.sprint('MBPT: setting up W(iw)',0)
            W = screened_interaction(st, pm.mbpt.screening, epsilon=eps)  # This is the screening interaction (W-v)
            save(W, "W{}_iw".format(cycle))
            save(W, "W_iw")
            del eps # not needed anymore

            pm.sprint('MBPT: transforming W to imaginary time',0)
            W = fft_t(W, st, dir='if2it', screening=pm.mbpt.screening)
            save(W, "W{}_it".format(cycle))
            save(W, "W_it")

        # compute Sc
        pm.sprint('MBPT: computing Sc(it)',0)
        Sc = self_energy_correlation(G, W, st, screening=pm.mbpt.screening)
        save(Sc, "Sc{}_it".format(cycle))
        save(Sc, "Sc_it")

        pm.sprint('MBPT: transforming Sc to imaginary frequency',0)
        Sc = fft_t(Sc, st, dir='it2if', screening=pm.mbpt.screening)
        save(Sc, "Sc{}_iw".format(cycle))
        save(Sc, "Sc_iw")

        # compute S
        S, Sx, Sxc = self_energy(pm, st, H, h0, Sc, W, pm.mbpt.screening)
        save(Sx, "Sx{}_iw".format(cycle))
        save(Sx, "Sx_iw")
        save(Sxc, "Sxc{}_iw".format(cycle))
        save(Sxc, "Sxc_iw")
        save(S, "S{}_iw".format(cycle))
        save(S, "S_iw")
        del Sx # not needed anymore
        del Sxc # not needed anymore
        del Sc # not needed anymore
        if not pm.mbpt.flavour == 'GW0':
            del W # not needed anymore

        cycle = cycle + 1
        pm.sprint('')
        pm.sprint('MBPT: Entering self-consistency cycle #{}'.format(cycle))

        # computing G
        pm.sprint('MBPT: solving the Dyson equation for new G',0)
        if pm.mbpt.screening == 'dynamic':
            G = solve_dyson_equation(G0, S, st, pm.mbpt.screening, pm=pm) # note: G0 = G0(r,r';iw)
        elif pm.mbpt.screening == 'static' or pm.mbpt.screening == 'inertial' or pm.mbpt.screening == 'zero':
            if cycle == 0:
                H.orbitals = copy.copy(h0.orbitals)
            G = solve_dyson_equation(G0, S, st, pm.mbpt.screening, pm=pm) # note: G0 = G0(r,r';iw)
        else:
            raise ValueError("Unrecognized screening {} for screened interaction".format(pm.mbpt.screening))

        pm.sprint('MBPT: transforming G to imaginary time',0)
        G = fft_t(G, st, dir='if2it')
        save(G, "G{}_it".format(cycle))
        save(G, "G_it")

        # extract density
        G_mzero = G[:,:,0]
        den_new = np.diagonal(G_mzero.imag).copy()
        results.add(den_new, "gs_mbpt_den{}".format(cycle))
        if pm.run.save:
            results.save(pm, list=["gs_mbpt_den{}".format(cycle)])
        den_norm = np.sum(den_new) * st.x_delta
        pm.sprint("MBPT: norm of new density: {:.3f} electrons".format(den_norm))
        den_maxdiff = np.max(np.abs(den_new - H.den))
        H.den = den_new

        # Compute quasiparticle energies and orbitals
        H.energies, H.orbitals, ip, af, gap = quasiparticle_orbitals(pm, st, S, h0, H, pm.mbpt.screening)

        # continue self-consitency?
        if pm.mbpt.flavour == 'G0W0':
            break
        elif cycle == pm.mbpt.max_iter:
            pm.sprint("Reached maximum number of iterations. Stopping...")
            break
        elif den_maxdiff < pm.mbpt.den_tol:
            pm.sprint("MBPT: convergence reached, exiting self-consistency cycle",0)
            break
        pm.sprint("MBPT: Max. change in den: {:.2e} > {:.2e}".format(den_maxdiff,pm.mbpt.den_tol))
        H.vh = hartree_potential(st, den=H.den)
        H.vx = -G_mzero.imag * st.coulomb_repulsion # = iGv

        # extrapolate G(it=0) from above
        eps = np.max(np.abs(G.real))
        if eps > pm.mbpt.den_tol:
            pm.sprint("MBPT: Warning: Discarding real part with max. {:.3e} during extrapolation".format(eps))
        G_pzero = extrapolate_to_zero(G, st, 'from_above')

    # normalise and save density and all objects
    den = H.den * st.NE / (np.sum(H.den) * st.x_delta)
    results.add(den, "gs_mbpt_den")
    results.add(H.vh, "gs_mbpt_vh")
    results.add(H.orbitals, "gs_mbpt_eigf") # if screening=dynamic will just return starting orbitals
    results.add(H.energies, "gs_mbpt_eigv")
    results.add(ip, "gs_mbpt_IP")
    results.add(af, "gs_mbpt_AF")
    results.add(gap, "gs_mbpt_GAP")
    l = ["gs_mbpt_den", "gs_mbpt_vh", "gs_mbpt_eigf", "gs_mbpt_eigv", "gs_mbpt_IP", "gs_mbpt_AF", "gs_mbpt_GAP"]
    if pm.run.save:
        results.save(pm, list=l)

    # TDGW still a work in progress
    # if pm.run.time_dependence:
    #
	# # check flavour
    #     if pm.mbpt.screening == 'dynamic':
    #         raise AttributeError('Error: Cannot perform TD-MBPT with dynamic screening, change pm.mbpt.screening to \'static\'')
    #
    #     # compute starting Hamiltonian
    #     sd = pm.space.second_derivative
    #     sd_ind = pm.space.second_derivative_indices
    #     K = -0.5*sps.diags(sd, sd_ind, shape=(pm.sys.grid,pm.sys.grid), format='csr', dtype=complex)
    #     Vext = sps.diags(pm.space.v_ext, 0, shape=(pm.sys.grid,pm.sys.grid), format='csr', dtype=complex).toarray()
    #     Vptrb = sps.diags(pm.space.v_pert, 0, shape=(pm.sys.grid,pm.sys.grid), format='csr', dtype=complex).toarray()
    #     Sc = Sc[:,:,0]
    #     H = K + Vext + Vptrb + H.vh + H.vx + Sc
    #
    #     # compute starting orbitals
    #     den, waves, eigv = iDEA.HF.groundstate(pm, H)
    #
    #     # initialise TD quantities
    #     n_t = np.empty((pm.sys.imax, pm.sys.grid), dtype=np.float)
    #     Vh_t = np.empty((pm.sys.imax, pm.sys.grid) , dtype=np.float)
    #     Sx_t = np.empty((pm.sys.imax, pm.sys.grid, pm.sys.grid), dtype=np.complex)
    #     Sc_t = np.empty((pm.sys.imax, pm.sys.grid, pm.sys.grid), dtype=np.complex)
    #     n_t[0] = den
    #     Vh_t[0,:] = H.vh
    #     Sx_t[0,:,:] = H.vx
    #     Sc_t[0,:,:] = Sc
    #
    #     # perfrom time-dependent iterations
    #     for i in range(1, pm.sys.imax):
    #         string = 'MBPT: evolving through real time: t = {:.4f}'.format(i*pm.sys.deltat)
    #         pm.sprint(string, 1, newline=False)
    #
    #         # perform a CN timestep
    #         waves = iDEA.HF.crank_nicolson_step(pm, waves, H)
    #         den = iDEA.HF.electron_density(pm, waves)
    #
    #         # compute new hamiltonian
    #         Sc = td_correlation(pm, waves)
    #         H, Vh, Sx, Sc = hamiltonian(pm, waves, den, i, Sc, perturb=True)
    #
    #         # record time dependent qunatities
    #         n_t[i] = den
    #         Vh_t[i,:] = Vh
    #         Sx_t[i,:,:] = Sx
    #         Sc_t[i,:,:] = Sc
    #
    #     # calculate the current density
    #     pm.sprint()
    #     current_density = calculate_current_density(pm, n_t)
    #
    #     # output results
    #     pm.sprint('MBPT: saving quantities...', 1, newline=True)
    #     results.add(n_t, 'td_mbpt_den')
    #     results.add(Vh_t, 'td_mbpt_vh')
    #     results.add(Sx_t, 'td_mbpt_Sx')
    #     results.add(Sc_t, 'td_mbpt_Sc')
    #     results.add(current_density, 'td_mbpt_cur')
    #
	# # save results
    #     if pm.run.save:
    #         l = ['td_mbpt_den','td_mbpt_cur', 'td_mbpt_Sx', 'td_mbpt_Sc', 'td_mbpt_vh']
    #         results.save(pm, list=l)

    return results


def read_input_quantities(pm, st):
    """Reads quantities of starting Hamiltonian h0

    This includes single-particle energies, orbitals and the density.

    .. math ::

        \mathcal{H}_0 = T + V_{ext}(r) + V_{Hxc}(r,r') \\
        V_{Hxc}(r,r') = \delta(r-r')V_H(r) + V_x(r,r') + V_c(r,r')

    Possible flavours of pm.mbpt.h0 are
      * 'lda1/2/3'/'ext': :math:`V_{Hxc}(r,r') = \delta(r-r') (V_H(r) + V_{xc}(r))`
         This form also applies to any reverse-engineered input.
      * 'h': :math:`V_{Hxc}(r,r') = \delta(r-r') V_H(r)`
      * 'hf': :math:`V_{Hxc}(r,r') = \delta(r-r') V_H(r) + V_x(r,r')`
      * 'non': :math:`V_{Hxc}(r,r') = 0`

    parameters
    ----------
    pm : object
        input parameters
    st : object
        space-time grid

    Returns
    -------
        Container object
    """
    flavour = pm.mbpt.h0

    energies = rs.Results.read('gs_{}_eigv'.format(flavour), pm)
    orbitals = rs.Results.read('gs_{}_eigf'.format(flavour), pm)
    den = rs.Results.read('gs_{}_den'.format(flavour), pm)

    if energies.dtype == np.complex:
        im_max = np.max(np.abs(energies.imag))
        s  = "MBPT: Warning: single-particle energies are complex (maximum "
        s += "imaginary component: {:.3e}). Casting to real. ".format(im_max)
        pm.sprint(s)
        energies = energies.real

    nener = len(energies)
    norb = len(orbitals)
    if nener != norb:
        raise ValueError("Number of starting orbitals {} doesn't equal number of starting energies {}".format(norb, nener))
    elif norb < st.norb:
        raise ValueError("Not enough orbitals: {} computed, {} requested.".format(norb,st.norb))
    else:
        energies = energies[:st.norb]
        orbitals = orbitals[:st.norb]

    # Shifting energies such that E=0 is half way between homo and lumo
    homo = energies[st.NE-1]
    lumo = energies[st.NE]
    gap = lumo - homo
    pm.sprint('MBPT: single-particle gap: {:.3f} Ha'.format(gap),0)
    e_fermi = (homo + lumo) / 2
    pm.sprint('MBPT: single-particle Fermi energy: {:.3f} Ha'.format(e_fermi),0)
    energies -= e_fermi

    # check that G(it) is well described
    exp_factor = np.exp(-(lumo-e_fermi)*st.tau_max)
    if exp_factor > 1e-1:
        t1 = -np.log(1e-1)/(lumo-e_fermi)
        t2 = -np.log(1e-2)/(lumo-e_fermi)
        s  = "MBPT: Warning: Width of tau-grid for G(it) is too small "
        s += "for HOMO-LUMO gap {:.3f} Ha. ".format(gap)
        s += "Increase tau_max to {:.1f} for decay to 10% ".format(t1)
        s +=  "or {:.1f} for decay to 1%".format(t2)
        pm.sprint(s)

    # computing & reading potentials
    vh = hartree_potential(st, den=den)
    vx = exchange_potential(st, orbitals=orbitals)
    vhxc = np.zeros((st.x_npt, st.x_npt), dtype=np.complex)
    if flavour == 'non':
        # non-interacting: v_Hxc = 0
        pass
    elif flavour == 'h':
        # Hartree: v_Hxc = v_H
        np.fill_diagonal(vhxc, vh / st.x_delta)
    elif flavour in ['lda1', 'lda2', 'lda3', 'ldaheg', 'nonre', 'hre', 'lda1re', 'lda2re', 'lda3re', 'ldahegre', 'extre', 'hfre']:
        # KS-DFT: v_Hxc = v_H + v_xc
        # (or any reverse-engineered starting point)
        tmp = vh + rs.Results.read('gs_{}_vxc'.format(flavour), pm)
        np.fill_diagonal(vhxc, tmp / st.x_delta)
    elif flavour == 'hf':
        # Hartree-Fock: v_Hxc = v_H + v_x
        np.fill_diagonal(vhxc, vh / st.x_delta)
        vhxc += vx
    else:
        raise ValueError("Unknown h0 flavour '{}'".format(flavour))
    h0 = Container()
    h0.energies = energies
    h0.orbitals = orbitals
    h0.den = den
    h0.e_fermi = e_fermi
    h0.vh = vh
    h0.vx = vx
    h0.vhxc = vhxc
    return h0


def hartree_potential(st, den=None, G=None):
    r"""Sets up Hartree potential V_H(r) from electron density.

    .. math::

       V_H(r) = \int \frac{\rho(r')}{|r-r'|} dr' = (-i) \int \frac{G(r',r';0)}{|r-r'|}dr'

    Note: :math:`V_H(r,r';i\tau) = \delta(r-r')\delta(i\tau)V_H(r)` with
    :math:`\delta(i\tau)=i\delta(\tau)`.

    parameters
    ----------
    st : object
       space-time grid
    den : array_like
       density
    G: array_like
        Green function G(r,r';it)

    Returns array_like
       vH(r)
    """
    if den is not None:
        pass
    elif G is not None:
        den = np.diagonal(G[:,:,0].imag).copy()
    else:
        raise IOError("Need to provide either den or G.")
    v_h = np.dot(st.coulomb_repulsion, den) * st.x_delta
    return v_h


def exchange_potential(st, G=None, orbitals=None):
    r"""Calculate Fock exchange operator V_x(r,r')

    Can take either the Green function G(it) as input or the orbitals.

    parameters
    ----------
    st : object
       space-time grid
    G : array_like
       Green function G(r,r';it) (or G(r,r';t))
    orbitals : array_like
       single-particle orbitals

    Returns array_like
        v_x(r,r')
    """
    if G is not None:
        # default multiplication is element-wise
        v_x = 1J * G[:,:,0] * st.coulomb_repulsion
    elif orbitals is not None:
        v_x = np.zeros((st.x_npt,st.x_npt),dtype=np.complex)
        for i in range(st.NE):
            orb = orbitals[i]
            v_x -= np.tensordot(orb.conj(), orb, axes=0)
        v_x = v_x * st.coulomb_repulsion
    else:
        raise IOError("Need to provide either G or orbitals.")
    return v_x


def self_screening_correction(st, den):
    r"""Evaluates our self-screening correction functinal for a given density.

    parameters
    ----------
    st : object
      contains space-time parameters
    den : array_like
      density

    Returns
    -------
    Vssc: array_like
      self-screening correction potential
    """
    a = 4.09268097
    b = 9.20608941
    c = 0.53651521
    return a*den*np.exp(-b*den**c)*(2.0-b*c*den**c)


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

      - '0+': :math:`G(0) = \lim_{t\downarrow 0}G(it)`,
        determined by empty states
      - '0-': :math:`G(0) = \lim_{t\uparrow 0}G(it)`,
        determined by occupied states with :math:`(-i)G(r,r,0)=\rho(r)`
      - 'both': return '0-' Green function *and* it=0 slice of '0+' Green function

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
    r"""Calculate expectation values of O(r,r';t) for each t wrt orbitals

    .. math:: O_{ij}(t) = \langle \varphi_i | O(r,r';t) | \varphi_j\rangle
                        = \int \varphi_i^*(r) O(r,r';t)\varphi_j(r')\,dr\,dr'

    Note: For non-hermitian O, the pairing of r,r' with
    :math":`\varphi_i^*,\varphi_j` does matter

    parameters
    ----------
    O: array
        Operator O(r,r';t)
    orbitals: array
        Array of shape (norb, grid) containing orbtials
    st: object
        space-time object
    mode: string
        - if 'diagonal': computes <j|O|j> for all j (default)
        - if 'full': computes <i|O|j> for all i,j

    Returns
    -------
    bracket_r: array_like
        - if mode == 'diagonal': bracket_r[i,k]
        - if mode == 'full': bracket_r[i,j,k]

        i,j orbital indices, k index of temporal grid
    """
    orbs = copy.copy(orbitals) * st.x_delta  # factor needed for integration
    # Performing one matrix-matrix multiplication is substantially faster than
    # performing t matrix-vector multiplications.
    # Note: If we do not reshape, np.dot internally performs matrix-vector
    # multiplications here.
    # numpy.dot(a,b) sums over last axis of a and 2nd-to-last axis of b
    # tmp.shape == (norb, x_npt*tau_npt)
    tmp = np.dot(orbs, O.reshape((st.x_npt, st.x_npt * st.tau_npt)))
    if mode == 'diagonal':
        # Then, we do element-wise multiplication + summation over the grid axis
        # to get the scalar product.
        bracket_r = (orbs.conj()[:,:,None] * tmp.reshape((st.norb,st.x_npt,st.tau_npt))).sum(1)
    elif mode == 'full':
        # bracket_r.shape == (norb,norb,t)
        bracket_r  = np.dot(orbs.conj(), tmp.reshape((st.norb,st.x_npt,st.tau_npt)))
    else:
        raise ValueError("Unknown mode {}".format(mode))
    return bracket_r


def fft_t(F, st, dir, screening='dynamic', phase_shift=False):
    r"""Performs 1d Fourier transform of F(r,r';t) along time dimension.

    Can handle forward & backward transforms in real & imaginary time.
    Here, we replicate the convention of [Rieger1999]_ (see equations 3.1 and 3.2)

    .. math::

       F(\omega) &= \int dt F(t) e^{i\omega t} \\
       F(t) &= \int \frac{d\omega}{2\pi} F(\omega) e^{-i\omega t}\\
       F(i\omega) &= -i\int dt F(it) e^{-i\omega t}\\
       F(it) &= i\int \frac{d\omega}{2\pi} F(i\omega) e^{i\omega t}

    The infinitesimals :math:`d\tau,d\omega/2\pi` are automatically
    included in the Fourier transforms.

    Note: We adopt the Fourier transform convention by Rieger, Hedin et al.,
    which uses negative imaginary exponents for the *backward* transform in real time.
    This differs from the more common convention (adopted by numpy) of using
    negative exponents for the *forward* transform.

    FLOPS: tau_npt * grid**2 * (log(grid) + 2)

    parameters
    ----------
    F: array
      will be transformed along last axis
    st: object
        space-time object
    dir: string
      - 't2f': time to frequency domain
      - 'f2t': frequency to time domain
      - 'it2if': imaginary time to imaginary frequency domain
      - 'if2it': imaginary frequency to imaginary time domain
    screening : string
        Use 'dynamic' (frequency dependent), 'static' (frequency independent averaged), 'inertial' (frequency independent at zero) or 'zero' no screening.
    phase_shift: bool
      - True: use with shifted tau grid (tau_grid[0] = tau_delta/2)
      - False: use with unshifted tau grid (tau_grid[0] = 0)
    """
    if screening == 'dynamic':
        n = float(F.shape[-1])
        # Crazy - taking out the prefactors p really makes it faster
        if dir == 't2f':
            out = ifft_1d(F) * st.tau_delta
            #out = np.fft.ifft(F, axis=-1) * n * st.tau_delta
            if phase_shift:
                out *= st.phase_backward
        elif dir == 'f2t':
            p = 1 / (n * st.tau_delta)
            if phase_shift:
                out = fft_1d(F * st.phase_forward) * p
            else:
                out = fft_1d(F) * p
                #out = np.fft.fft(F, axis=-1) / (n * st.tau_delta)
        elif dir == 'it2if':
            p = -1J * st.tau_delta
            out = fft_1d(F) * p
            #out = -1J * np.fft.fft(F, axis=-1) * st.tau_delta
            if phase_shift:
                out *= st.phase_forward
        elif dir == 'if2it':
            p = 1J / (n * st.tau_delta)
            if phase_shift:
                out = ifft_1d(F * st.phase_backward) * p
            else:
                out = ifft_1d(F) * p
                #out = 1J * np.fft.ifft(F, axis=-1) / st.tau_delta
        else:
            raise IOError("FFT direction {} not recognized.".format(dir))
    elif screening == 'inertial' or screening == 'static' or screening == 'zero':
        if dir == 'it2if':
            out = -1J * F * st.tau_delta
        elif dir == 'if2it':
            out = 1J * F / st.tau_delta
        else:
            raise IOError("FFT direction {} not recognized.".format(dir))
    else:
        raise ValueError("Unrecognized screening {} for screened interaction".format(screening))
    return out


def irreducible_polarizability(st, G, G_pzero, screening):
    r"""Calculates irreducible polarizability P(r,r',it).

    .. math:: P(r,r';i\tau) = -iG(r,r';i\tau) G(r',r;-i\tau)

    parameters
    ----------
    G : array
        Green function
    G_pzero : array
        it=0 component of Green function with :math:`G(0) = \lim_{t\downarrow 0}G(it)`
    screening : string
        Use 'dynamic' (frequency dependent), 'static' (frequency independent averaged), 'inertial' (frequency independent at zero) or 'zero' no screening.

    FLOPS: grid**2 * tau_npt * 3

    See equation 3.4 of [Rieger1999]_.
    """
    if screening == 'zero':
        P = np.zeros((st.x_npt,st.x_npt), dtype=np.complex)
    elif screening == 'dynamic' or screening == 'static':
        G_rev = copy.copy(G)
        G_rev = G_rev.swapaxes(0,1)
        G_rev[:,:,0] = G_pzero
        # need t=0 to become the *last* index for ::-1
        G_rev = np.roll(G_rev, -1, axis=2)
        P =  -1J * G * G_rev[:,:,::-1]
    elif screening == 'inertial':
        P =  -1J * G[:,:,0] * G_pzero
        P = P / st.tau_delta
    else:
        raise ValueError("Unrecognized screening {} for screened interaction".format(screening))
    return P


def dielectric_matrix(P, screening, st):
    r"""Calculates dielectric matrix eps(r,r';iw) from polarizability.

    .. math::

        \varepsilon(r,r';i\omega) = \delta(r-r') - \int v(r-r'')P(r'',r';i\omega)d^3r''

    See equation 3.5 and 4.3 of [Rieger1999]_.

    FLOPS: tau_npt * (2*grid * grid**2)

    parameters
    ----------
    P: array_like
      irreducible polarizability P(r,r';iw)
    screening : string
        Use 'dynamic' (frequency dependent), 'static' (frequency independent averaged), 'inertial' (frequency independent at zero) or 'zero' no screening.
    st: object
      space-time grid
    """
    v = st.coulomb_repulsion
    if screening == 'zero':
        eps = np.zeros((st.x_npt,st.x_npt), dtype=np.complex)
        tmp = 1.0 / st.x_delta
        for i in range(st.x_npt):
            eps[i, i] += tmp
    elif screening == 'dynamic' or screening == 'static':
        # Note: dot + reshape is a tiny bit faster than tensordot...
        #eps = - np.tensordot(v, P, axes=(1,1)) * st.x_delta
        eps = - np.dot(v, P.reshape(st.x_npt, st.x_npt*st.tau_npt)) * st.x_delta
        eps = eps.reshape(st.x_npt, st.x_npt, st.tau_npt)
        # add delta(r-r')
        tmp = 1.0 / st.x_delta
        for i in range(st.x_npt):
            eps[i, i, :] += tmp
    elif screening == 'inertial':
        eps = np.zeros((st.x_npt,st.x_npt), dtype=np.complex)
        eps[:, :] = -np.dot(v, P[:, :]) * st.x_delta
        # add delta(r-r')
        for i in range(st.x_npt):
            eps[i, i] += 1 / st.x_delta
    else:
        raise ValueError("Unrecognized screening {} for screened interaction".format(screening))
    return eps


def screened_interaction(st, screening, epsilon):
    r"""Calculates screened interaction W(r,r';iw).

    Input is the (inverse) dielectric function.

    .. math::

        W(r,r';i\omega) = \int v(r-r'') \varepsilon^{-1}(r'',r';i\omega)d^3r''

    See equation 3.6 of [Rieger1999]_.

    FLOPS: tau_npt * (grid**3 + grid**2)

    parameters
    ----------
    st: object
        space-time object
    screening : string
        Use 'dynamic' (frequency dependent), 'static' (frequency independent averaged), 'inertial' (frequency independent at zero) or 'zero' no screening.
    epsilon: array_like
        dielectric matrix eps(r,r',iw).
        If provided, we solve epsilon W = v instead

    Returns
    -------
        screened interaction W
    """
    if screening == 'zero':
        W = np.zeros((st.x_npt, st.x_npt), dtype=complex)
    elif screening == 'dynamic':
        W = np.empty((st.x_npt, st.x_npt, st.tau_npt), dtype=complex)
        v = st.coulomb_repulsion
        # solve eps*(W+v) = v/dx
        v_dx = v / st.x_delta
        for k in range(st.tau_npt):
            W[:, :, k] = np.linalg.solve(epsilon[:,:,k], v_dx) - v # Dynamic screening W(iw)=e^-1(iw)*v
    elif screening == 'static':
        W = np.empty((st.x_npt, st.x_npt), dtype=complex)
        v = st.coulomb_repulsion
        # solve eps*(W+v) = v/dx
        v_dx = v / st.x_delta
        W[:, :] = np.linalg.solve(epsilon[:,:,0], v_dx) - v # Static screening W=e^-1(iw=0)*v
    elif screening == 'inertial':
        W = np.empty((st.x_npt, st.x_npt), dtype=complex)
        v = st.coulomb_repulsion
        # solve eps*(W+v) = v/dx
        v_dx = v / st.x_delta
        W[:, :] = np.linalg.solve(epsilon[:,:], v_dx) - v # Inertial screening W=e_inertial^-1*v
    else:
        raise ValueError("Unrecognized screening {} for screened interaction".format(screening))
    return W


def self_energy_correlation(G, W, st, screening):
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
    st: object
        space-time object
    screening : string
        Use 'dynamic' (frequency dependent), 'static' (frequency independent averaged), 'inertial' (frequency independent at zero) or 'zero' no screening.

    return S
    """
    if screening == 'zero':
        Sc = np.zeros((st.x_npt, st.x_npt), dtype=complex)
    elif screening == 'dynamic':
        Sc = 1J * G * W
    elif screening == 'static':
        Sc = 1J * G[:,:,0] * W
    elif screening == 'inertial':
        Sc = 1J * G[:,:,0] * W
    else:
        raise ValueError("Unrecognized screening {} for screened interaction".format(screening))
    return Sc


def self_energy(pm, st, H, h0, Sc, W, screening):
    r"""Calculate the full self-energy S(iw) within the GW approximation.

    parameters
    ----------
    pm: object
       parameters object
    st: object
       space-time object
    H: object
       interacting Hamiltonian
    h0: object
       non-interacting Hamiltonain
    G: array_like
       Green function G(it)
    W: array_like
       Screened interaction W(it)
    screening : string
        Use 'dynamic' (frequency dependent), 'static' (frequency independent averaged), 'inertial' (frequency independent at zero) or 'zero' no screening.

    return S
    """
    # compute Sx
    if screening == 'dynamic':
        Sx = np.zeros(Sc.shape, dtype=np.complex)
        for i in range(st.tau_npt):
            Sx[:,:,i] = H.vx
    elif screening == 'static' or screening == 'inertial' or screening == 'zero':
        Sx = np.zeros(Sc.shape, dtype=np.complex)
        Sx[:,:] = H.vx
    else:
        raise ValueError("Unrecognized screening {} for screened interaction".format(screening))

    # compute Sxc = Sx + Sc
    Sxc = Sx + Sc

    # construct the full S
    if screening == 'dynamic':
        S = np.zeros(Sc.shape, dtype=np.complex)
        pm.sprint('MBPT: constructing S(iw)',0)
        Vh = np.diag(H.vh) / st.x_delta
        for i in range(st.tau_npt):
            S[:,:,i] = Vh[:,:] + Sxc[:,:,i] - h0.vhxc[:,:]  # S = Vh + Sxc - Vhxc0
    elif screening == 'static':
        S = np.zeros(Sc.shape, dtype=np.complex)
        pm.sprint('MBPT: constructing S(iw)',0)
        Vh = np.diag(H.vh) / st.x_delta
        W = fft_t(W, st, dir='it2if', screening=pm.mbpt.screening)
        COH = 0.5*np.diag(np.diag(W)) / st.x_delta # COH term in the static approximation
        S = (Vh + Sxc + COH)*st.x_delta
    elif screening == 'inertial':
        S = np.zeros(Sc.shape, dtype=np.complex)
        pm.sprint('MBPT: constructing S(iw)',0)
        Vh = np.diag(H.vh) / st.x_delta
        S = (Vh + Sxc)*st.x_delta
    elif screening == 'zero':
        S = np.zeros(Sc.shape, dtype=np.complex)
        pm.sprint('MBPT: constructing S(iw)',0)
        Vh = np.diag(H.vh) / st.x_delta
        S = (Vh + Sxc)*st.x_delta
    else:
        raise ValueError("Unrecognized screening {} for screened interaction".format(screening))

    # remove the self-screening error from S
    if(pm.mbpt.ssc == True):
        pm.sprint('MBPT: removing the self-screening error from sigma')
        den = H.den * st.NE / (np.sum(H.den) * st.x_delta)
        vssc = self_screening_correction(st, abs(den))
        vssc_matrix = np.diag(vssc) / st.x_delta
        if screening == 'dynamic':
            for i in range(st.tau_npt):
                S[:,:,i] = S[:,:,i] + vssc_matrix
        elif screening == 'static' or screening == 'inertial' or screening == 'zero':
            S[:,:] = S[:,:] + vssc_matrix
        else:
            raise ValueError("Unrecognized screening {} for screened interaction".format(screening))

    # Align fermi energy of input and output Green function
    if pm.mbpt.hedin_shift and screening == 'dynamic':
        pm.sprint('MBPT: performing Hedin shift',0)

        # Get QP energies
        H.sigma_iw_dg = bracket_r(S, h0.orbitals, st)
        qp_energies = H.sigma_iw_dg[:,0].real + h0.energies

        # Sort orbitals and energies if required
        if not all(qp_energies[i] <= qp_energies[i+1] for i in range(0,len(qp_energies)-2)):
            pm.sprint("MBPT: Warning: QP energies out of order, reordering the following indices...")
            indices = np.argsort(qp_energies)
            pm.sprint("MBPT: {}".format(indices))
            qp_energies = qp_energies[indices]
            h0.orbitals = h0.orbitals[indices]
            h0.energies = h0.energies[indices]

        # Do Hedin Shift
        H.qp_shift = 0.5 * (qp_energies[st.NE-1] + qp_energies[st.NE]) # Quasi-particle shift to keep fermi-energy in HOMO-LUMO gap
        H.qp_shift = H.qp_shift.real # Take just the real part
        pm.sprint('MBPT: quasi-particle shift: {:.7f} Ha.'.format(H.qp_shift))
        for i in range(st.x_npt):
            S[i,i,:] -= H.qp_shift / st.x_delta # Perfrom hedin shift

        # Print new qp_energies
        H.sigma_iw_dg = bracket_r(S, h0.orbitals, st)[:,0].real + h0.energies
        pm.sprint('MBPT: qp_energies after shift: {}...'.format(H.sigma_iw_dg[0:pm.sys.NE+2]),0)

    return S, Sx, Sxc


def solve_dyson_equation(G0, S, st, screening, pm=None, orbitals=None):
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
    screening : string
        Use 'dynamic' (frequency dependent), 'static' (frequency independent averaged), 'inertial' (frequency independent at zero) or 'zero' no screening.

    Returns
    -------
        updated Green function G(r,r';iw)
    """
    if screening == 'dynamic':
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
        # using explicit matix inversion: significantly slower but equivalent to the above code
        #A = inverse_r(A, st)
        #for k in range(st.tau_npt):
        #    G[:,:,k] = np.dot(A[:,:,k], G0[:,:,k]) * st.x_delta
    elif screening == 'zero' or screening == 'static' or screening == 'inertial':
        H_s = hamiltonian(pm, st, S)
        den, orbitals, energies = iDEA.HF.groundstate(pm, H_s)
        e_fermi = (energies[st.NE-1] + energies[st.NE]) / 2
        energies -= e_fermi
        G = non_interacting_green_function(-orbitals.T[:st.norb], energies[:st.norb], st)
        G = fft_t(G, st, dir='it2if')
    else:
        raise ValueError("Unrecognized screening {} for screened interaction".format(screening))
    return G


def hamiltonian(pm, st, S):
    r"""Calculates Hamiltonian from a static correlation self-energy.

    .. math ::

        H=K+V_{ext}+V_{H}+\Sigma_{x}+\Sigma_{c}

    parameters
    ----------
    pm : array
      set of single-particle orbitals
    st: object
      space-time object
    S: self-energy

    """
    # construct kinetic energy
    sd = pm.space.second_derivative
    sd_ind = pm.space.second_derivative_indices
    K = -0.5*sps.diags(sd, sd_ind, shape=(pm.sys.grid,pm.sys.grid), format='csr', dtype=complex)
    # construct H from individual terms
    Vext = sps.diags(pm.space.v_ext, 0, shape=(pm.sys.grid,pm.sys.grid), format='csr', dtype=complex).toarray()
    H = K + Vext + S
    return H


def quasiparticle_orbitals(pm, st, S, h0, H, screening):
    if screening == 'dynamic':
        d = np.gradient(S, st.omega_delta*1.0j, axis=-1) # dS/dw (derivative of sigma wrt imaginary frequency)
        qp_energies = h0.energies + bracket_r(S, h0.orbitals, st)[:,0].real +  H.qp_shift # Use 1st order approx to QP energies
        qp_energies = qp_energies + h0.e_fermi # Take account of the vaccum shift
        homo = qp_energies[st.NE-1]
        ip = -homo
        lumo = qp_energies[st.NE]
        af = -lumo
        gap = ip - af
        pm.sprint('MBPT: IP, AF, GAP: {0:.3f}, {1:.3f}, {2:.3f} Ha'.format(ip, af, gap))
        qp_orbitals = copy.copy(h0.orbitals)
    elif screening == 'static' or screening == 'inertial' or screening == 'zero':
        H_s = hamiltonian(pm, st, S)
        den, orbitals, energies = iDEA.HF.groundstate(pm, H_s)
        qp_energies = copy.copy(energies)
        homo = qp_energies[st.NE-1]
        ip = -homo
        lumo = qp_energies[st.NE]
        af = -lumo
        gap = ip - af
        qp_orbitals = copy.copy(orbitals.T)
        pm.sprint('MBPT: IP, AF, GAP: {0:.3f}, {1:.3f}, {2:.3f} Ha'.format(ip, af, gap))
    else:
        raise ValueError("Unrecognized screening {} for screened interaction".format(screening))
    return qp_energies, qp_orbitals, ip, af, gap


def extrapolate_to_zero(F, st, dir='from_below', order=6, points=7):
    """Extrapolate F(r,r';it) to it=0

    Note: Only the imaginary part is extrapolated.

    parameters
    ----------
    F: array_like
      quantity to extrapolate
    dir: string
      - 'from_below': extrapolate from negative imaginary times (default)
      - 'from_above': extrapolate from positive imaginary times
    order: int
      order of polynomial fit (order+1 parameters)
    points: int
      choose points=order+1 unless you face instability issues

    Returns
    -------
        extrapolated value F(r,r';it=0)
    """
    if dir == 'from_below':
        istart = st.tau_npt - points
        iend = st.tau_npt
    elif dir == 'from_above':
        istart = 1
        iend = 1 + points
    x = st.tau_grid[istart:iend]
    y = F[:,:, istart:iend].imag.reshape((st.x_npt*st.x_npt,points))
    coefs = np.polynomial.polynomial.polyfit(x,y.T, order)
    vals = np.polynomial.polynomial.polyval(0,coefs)
    vals = vals.reshape((st.x_npt, st.x_npt)) * 1J

    return vals

# def td_correlation(pm, waves, st):
#     # calculate P
#     if pm.mbpt.screening == 'zero':
#         P = np.zeros((st.x_npt, st.x_npt), dtype=np.complex)
#     elif pm.mbpt.screening == 'static':
#         a = np.zeros((st.x_npt, st.x_npt), dtype=np.complex)
#         b = np.zeros((st.x_npt, st.x_npt), dtype=np.complex)
#         for i in range(0, pm.sys.NE):
#             wave = waves[:,i]
#             a += np.tensordot(wave.conj(), wave, axes=0)
#         for j in range(pm.sys.NE, norb):
#             wave = waves[:,j]
#             b += np.tensordot(wave, wave.conj(), axes=0)
#         P = -1.0j*a*b # maybe should be P = 1.0j*a*b ('-' removed via FT?)
#     elif pm.mbpt.screening == 'dynamic':
#         raise AttributeError('Error: Cannot perform TD-MBPT with dynamic screening, change pm.mbpt.screening to \'static\'')
#     else:
#         raise ValueError("Unrecognized screening {} for screened interaction".format(screening))
#     # calculate W
#     dx = st.x_delta
#     I = np.identity(st.x_npt)
#     v = st.coulomb_repulsion
#     W = np.dot(np.inv(I/dx - np.dot(v,P)*dx),v)*dx # maybe should be (1/2*pi) (added via FT?)
#     # calculate Sc
#     Sc = -a*W # maybe should be (a*W) ('-' removed via FT?)
#     return Sc


# def analytic_continuation(S, st, pm, n_poles=3, fit_range=None):
#     r"""Fit poles to self-energy along the imaginary frequency axis
#
#     .. math::
#
#         \langle \varphi_j | \Sigma(z) | \varphi_j\rangle = a_j^0 + \sum_{k=1}^n \frac{a_j^k}{b_j^k-z}
#
#     Since :math:`\Sigma(i\omega)=\Sigma(-i\omega)^*`, it is sufficient to fit
#     half of the imaginary axis. Note that fitting the while imaginary axis won't
#     work because our fitting function do *not* share this property.
#
#     parameters
#     ----------
#     S: array_like
#       Matrix elements <i|sigma(iw)|j> (which basis is irrelevant)
#     n_poles: int
#      Number of poles, i.e. n = n_poles + 1
#     fit_range: float
#       - if > 0: fit is restricted to frequencies [0, 1J*fit_range]
#       - if < 0: fit is restricted to frequencies [1J*fit_range, 0]
#
#     Returns
#     -------
#       Polefit object, which contains the functions fitted to the
#       respective matrix elements of the self-energy
#
#     See equation 5.1 in [Rieger1999]_.
#     """
#
#     if fit_range is None:
#         # by default, we fit the upper imaginary axis
#         fit_range = st.omega_max
#
#     if fit_range > 0:
#         fit_half_plane = 'upper'
#         # need bitwise and here
#         select = (st.omega_grid >= 0) & (st.omega_grid <= fit_range)
#     else:
#         fit_half_plane = 'lower'
#         select = (st.omega_grid <= 0) & (st.omega_grid >= -fit_range)
#
#     # note: np.where(select).shape is (1,len(indices))
#     indices = np.where(select)[0]
#     fit_omega_grid = st.omega_grid[indices]
#     fit_sigma_iw = S[:,indices]
#
#     n_states = len(S)
#     fits = []
#     for i_state in range(n_states):
#         #s = fit_sigma_iw[i_state]
#         #popt, pconv = so.curve_fit(f, st.tau_grid, S)
#
#         x_fit = 1J*fit_omega_grid
#         y_fit = fit_sigma_iw[i_state]
#         fit = continuation.Polefit(n_poles=n_poles, fit_half_plane=fit_half_plane)
#         fit.fit(x=x_fit, y=y_fit)
#
#         bad_poles = fit.check_poles()
#         if bad_poles.size != 0:
#             msg = "Warning: State {}: Poles {} lie in {} half plane.".format(i_state,bad_poles,fit_half_plane)
#             pm.sprint(msg)
#
#         #p0 = cplx2float(np.array([0.1 for _i in range(2*n_poles+1)]))
#         # x = 1J*st.omega_grid # fitting on upper imaginary axis
#         #y = s
#         #p, cov = so.leastsq(residuals,p0,args=(x, y))
#         #p = float2cplx(p)
#
#         diff = y_fit - fit.f(1J * st.omega_grid[indices])
#         err = np.sum(np.abs(diff)**2) * st.omega_delta
#         #pm.sprint(err)
#         # Note: here, we would have the option to simply repeat the fit
#         if err > 1e-5:
#             pm.sprint("Warning: {}-pole fit differs by {:.1e} for state {}"
#                   .format(n_poles, err, i_state))
#
#         fits.append(fit)
#
#     return fits

# def optimise_hamiltonian(h0, S_w, st):
#     """Compute optimal non-interacting Hamiltonian for self-energy
#
#     This is quasi-particle self-consistent GW
#     """
#
#     h_opt = np.empty((st.norb,st.norb), dtype=float)
#
#     # need hartree energies of sp_orbitals for new rho
#     h_energies = hartree_energies(pm, orbitals=qp_orbitals, rho=rho)
#     #print(sigma_fits.shape)
#     #print(qp_energies.shape)
#     for i in range(st.norb):
#         for j in range(st.norb):
#             # mode A
#             # note: one could also use the *previous* qp_energies here
#             # in this case, there is no need to solve_for_qp_energies
#             # TODO: check whether this is more stable
#             h_opt[i,j] = 0.5*( S_w[i,j].f(qp_energies[i])\
#                            + S_w[i,j].f(qp_energies[j]))
#
#         #h_opt[i,i] += qp_energies[i].real
#         h_opt[i,i] += h0.sp_energies[i] + hartree_energies
#         h_opt[i,i] -= H.qp_energies[i].real
#
#     # take Hermitian part
#     h_opt = 0.5 * (h_opt + h_opt.H)
#
#     eigv, eigvec = la.eigh(h_opt)
#     qp_energies = eigv
#     # normalization in r-space: 1/sqrt(deltax)
#     # np.dot(a,b) sums over last axis of a and 2nd-to-last axis of b
#     eigvec = eigvec.swapaxes(0,1)
#     qp_orbitals = np.dot(eigvec, h0.orbitals)
