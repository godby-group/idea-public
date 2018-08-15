"""Direct minimisation of the Hamiltonian
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import scipy.linalg as spla

machine_epsilon = np.finfo(float).eps


class CGMinimizer(object):
    """Performs conjugate gradient minimization

    Performs Pulay mixing with Kerker preconditioner,
    as described on pp 1071 of [Payne1992]_
    """

    def __init__(self, pm, total_energy=None, nstates=None, cg_restart=3, line_fit='quadratic'):
        """Initializes variables

        parameters
        ----------
        pm: object
          input parameters
        total_energy: callable
          function f(pm, waves) that returns total energy
        nstates: int
          how many states to retain.
          currently, this must equal the number of occupied states
          (for unoccupied states need to re-diagonalize)
        cg_restart: int
          cg history is restarted every cg_restart steps
        line_fit: str
          select method for line-search
          'quadratic' (default), 'quadratic-der' or 'trigonometric'

        """
        self.pm = pm
        self.dx = pm.sys.deltax
        self.sqdx = np.sqrt(self.dx)
        self.dx2 = (self.dx)**2

        self.NE = pm.sys.NE
        if nstates is None:
            ## this is CASTEP's rule based on free-electron arguments
            ## + some heuristics
            #n_add = int(2 * np.sqrt(pm.sys.NE))
            #n_add = np.minimum(4, n_add)
            #self.nstates = pm.sys.NE + n_add
            self.nstates = pm.sys.NE
        else:
            self.nstates = nstates

        self.cg_dirs = np.zeros((pm.sys.grid, self.nstates))
        self.steepest_prods = np.ones((self.nstates))

        if total_energy is None:
            raise ValueError("Need to provide total_energy function that computes total energy from given set of single-particle wave functions.")
        else:
            self._total_energy = total_energy

        self.cg_restart = cg_restart
        self.cg_counter = 1

        self.line_fit = line_fit

    def exact_dirs(self, wfs, H):
        r"""Search direction from exact diagonalization

        Just for testing purposes (you can easily end up with
        multiple minima along the exact search directions)
        """

        eigv, eigf = spla.eigh(H)

        dirs = eigf[:,:self.nstates] - wfs
        overlaps = np.dot(wfs.conj().T, dirs)
        dirs = dirs - np.dot(wfs, overlaps)

        return dirs


    def step(self, wfs, H):
        r"""Performs one cg step

        After each step, the Hamiltonian should be recomputed using the updated
        wave functions.
        
        Note that we currently don't enforce the wave functions to remain
        eigenfunctions of the Hamiltonian. This should not matter for the total
        energy but means we need to performa a diagonalisation at the very end.

        parameters
        ----------
        wfs: array_like
          (grid, nwf) input wave functions
        H: array_like
          input Hamiltonian

        returns
        -------
        wfs: array_like
          (grid, nwf) updated wave functions
        """
        # internally work with dx=1
        wfs *= self.sqdx
        wfs = wfs[:, :self.nstates]

        #wfs = orthonormalize(wfs)
        # subspace diagonalisation also makes sure that wfs  are orthonormal
        wfs = self.subspace_diagonalization(wfs,H)

        exact = False
        if exact:
            conjugate_dirs = self.exact_dirs(wfs, H)
        else:
            steepest_dirs = self.steepest_dirs(wfs, H)
            conjugate_dirs = self.conjugate_directions(steepest_dirs, wfs)

        wfs_new = self.line_search(wfs, conjugate_dirs, H, mode=self.line_fit)

        return wfs_new / self.sqdx

    def line_search(self, wfs, dirs, H, mode):
        r"""Performs a line search along cg direction

        Trying to minimize total energy

        .. math ::
            E(s) = \langle \psi(s) | H[\psi(s)] | \psi(s) \rangle
        """

        E_0 = self.total_energy(wfs)
        dE_ds_0 = 2 * np.sum(self.braket(dirs[:,:self.NE], H, wfs[:,:self.NE]).real)

        if dE_ds_0 > 0:
                s = "CG: Warning: Energy derivative along conjugate gradient direction"
                s += "\nis positive: dE/dlambda = {:.3e}".format(dE_ds_0)
                self.pm.sprint(s)
                #raise ValueError(s)


        # Use only E_0 and dE_ds_0, assume constant H
        # This is useful, if the energy landscape is flat up to machine precision
        if mode == 'quadratic-der':
            H_dirs = np.dot(H, dirs)
            a = (wfs.conj() * H_dirs).sum()
            b = (dirs.conj() * H_dirs).sum()
            s_opt = -a/b
            wfs_new = wfs + s_opt * dirs

        # Use E_0, dE_ds_0 and E_1 + perform parabolic fit
        elif mode == 'quadratic':
            s_1 = 0.7 
            new_wfs = lambda s: wfs + dirs * s

            b = lambda s: dE_ds_0 * s
            a = lambda E_1, s: (E_1 - E_0 - b(s))/ s**2
            s_min = lambda E_1, s: -0.5 * b(s) / (a(E_1,s) * s)

            wfs_1 = new_wfs(s_1)
            wfs_1 = orthonormalize(wfs_1)
            E_1 = self.total_energy(wfs_1)

            eval_max = 10
            epsilon = 10 *  machine_epsilon * np.abs(E_0)
            for i in range(eval_max):
                if E_1 < E_0 + epsilon:
                    break
                else:
                    s_1 /= 2.0
                    #print(s_1)

                wfs_1 = new_wfs(s_1)
                wfs_1 = orthonormalize(wfs_1)
                E_1 = self.total_energy(wfs_1)
                i += 1

            # if energy landscape is just noise,
            # rely on gradient information only
            if np.abs(E_0 - E_1) < epsilon: # and np.abs(dE_ds_0) < epsilon:
                # this is non-selfconsistent for the moment
                # (would need to recompute H from wfs_1 otherwise)
                dE_ds_1 = 2 * np.sum(self.braket(dirs[:,:self.NE], H, wfs_1[:,:self.NE]).real)
                if dE_ds_1 < 0:
                        s = "CG: Warning: Energy derivative along conjugate gradient direction"
                        s += "\n at E1 is negative: dE/dlambda = {:.3e}".format(dE_ds_1)
                        self.pm.sprint(s)
                s_opt = - dE_ds_0/(dE_ds_1 - dE_ds_0) * s_1 
                #dE_ds_1 = 
                #H_dirs = np.dot(H, dirs)
                #a = (wfs.conj() * H_dirs).sum()
                #b = (dirs.conj() * H_dirs).sum()
                #s_opt = -a/b
                #dE_ds_1 = 2 * np.sum(self.braket(dirs[:,:self.NE], H, wfs_1[:,:self.NE]).real)

                print(s_1)
                print(s_opt)

            # else use energy E_1
            else:
                # eqn (5.30)
                s_opt = s_min(E_1, s_1)


            # we don't want the step to get too large
            s_opt = np.minimum(1.5,s_opt) 
            #s_opt = np.maximum(0,s_opt)

            if s_opt < 0:
                self.pm.sprint("CG: Warning: s_opt = {:.3e} < 0".format(s_opt))

            wfs_new = new_wfs(s_opt)

        # [Payne1992]_ method for (single-band) cg
        # Not sure this is really appropriate here
        # note: Payne et al. actually normalize the cg direction (?)
        elif mode == 'trigonometric':
            s_1 = np.pi/30
            wfs_1 = wfs * np.cos(s_1) + dirs * np.sin(s_1)
            wfs_1 = orthonormalize(wfs_1)
            E_1 = self.total_energy(wfs_1)

            # eqns (5.28-5.29)
            B_1 = 0.5 * dE_ds_0
            A_1 = (E_0 - E_1 + B_1)/ (1 - np.cos(2*s_1))
            s_opt = 0.5 * np.arctan(B_1/ A_1)

            # we don't want the step to get too large
            s_opt = np.minimum(1.5,s_opt) 
            wfs_new = wfs * np.cos(s_opt) + dirs * np.sin(s_opt)

        else:
            raise ValueError("CG: Unrecognised line-search mode '{}'".format(mode))


        wfs_new = orthonormalize(wfs_new)

        return wfs_new


    def braket(self, bra=None, O=None, ket=None):
        r"""Compute braket with operator O

        bra and ket may hold multiple vectors or may be empty.
        Variants:

        .. math:
            \lambda_i = \langle \psi_i | O | \psi_i \rangle
            \varphi_i = O | \psi_i \rangle
            \varphi_i = <psi_i | O

        parameters
        ----------
        bra: array_like
          (grid, nwf) lhs of braket
        O: array_like
          (grid, grid) operator. defaults to identity matrix
        ket: array_like
          (grid, nwf) rhs of braket
        """
        if O is None:
            if bra is None:
                return ket
            elif ket is None:
                return bra.conj().T
            else:
                return np.dot(bra.conj().T, ket)
        else:
            if bra is None:
                return np.dot(O, ket)
            elif ket is None:
                return np.dot(bra.conj().T, O)
            else:
                O_ket = np.dot(O, ket)
                return (bra.conj() * O_ket).sum(0)


    def steepest_dirs(self, wfs, H):
        r"""Compute steepest descent directions

        Compute steepest descent directions and project out components pointing
        along other orbitals
        (equivalent to steepest descent with the proper Lagrange multipliers).

        .. math:
            \zeta^{'m}_i = -(H \psi_i^m + \sum_j \langle \psi_j^m | H | \psi_i^m\rangle \psi_j^m

        See eqns (5.10), (5.12) in [Payne1992]_

        parameters
        ----------
        H: array_like
          Hamiltonian matrix (grid,grid)
        wavefunctions: array_like
          wave function array (grid, nwf)

        returns
        -------
        steepest_orth: array_like
          steepest descent directions (grid, nwf)
        """
        nwf = wfs.shape[-1]

        # \zeta_i = - H v_i
        steepest = - self.braket(None, H, wfs)
        # overlaps = (iwf, idir)
        overlaps = np.dot(wfs.conj().T, steepest)

        # \zeta_i += \sum_j <v_j|H|v_i> v_j
        steepest_orth = steepest - np.dot(wfs,overlaps)

        return steepest_orth

    def conjugate_directions(self, steepest_dirs, wfs):
        r"""Compute conjugate gradient descent for one state

        Updates internal arrays accordingly

        .. math:
            d^m = g^m + \gamma^m d^{m-1}
            \gamma^m = \frac{g^m\cdot g^m}{g^{m-1}\cdot g^{m-1}}

        See eqns (5.8-9) in [Payne1992]_

        parameters
        ----------
        steepest_dirs: array_like
          steepest-descent directions (grid, nwf)
        wfs: array_like
          wave functions (grid, nwf)

        returns
        -------
        cg_dirs: array_like
          conjugate directions (grid, nwf)
        """

        steepest_prods = np.linalg.norm(steepest_dirs)**2
        gamma = np.sum(steepest_prods)/ np.sum(self.steepest_prods)
        #gamma = steepest_prods / self.steepest_prods
        self.steepest_prods = steepest_prods

        if self.cg_counter == self.cg_restart:
            self.cg_dirs[:] = 0
            self.cg_counter = 0

        # cg_dirs = (grid, nwf)
        #cg_dirs = steepest_dirs + np.dot(gamma, self.cg_dirs)
        cg_dirs = steepest_dirs + gamma * self.cg_dirs
        #print(gamma)

        # orthogonalize to wfs vector
        # note that wfs vector is normalised to #electrons!
        #cg_dirs = cg_dirs - np.sum(np.dot(cg_dirs.conj(), wfs.T))/self.nstates * wfs
        # overlaps: 1st index wf, 2nd index cg dir
        overlaps = np.dot(wfs.conj().T, cg_dirs)
        cg_dirs = cg_dirs - np.dot(wfs, overlaps)
        self.cg_dirs = cg_dirs

        self.cg_counter += 1

        return cg_dirs


    def total_energy(self, wfs):
        r"""Compute total energy for given wave function

        This method must be provided by the calling module
        and is initialized in the constructor.
        """
        return self._total_energy(self.pm, wfs/self.sqdx)
  

    def subspace_diagonalization(self, v, H):
        """Diagonalise suspace of wfs
         
        parameters
        ----------
        v: array_like
          (grid, nwf) array of orthonormal vectors
        H: array_like
          (grid,grid) Hamiltonian matrix

        returns
        -------
        v_rot: array_like
          (grid, nwf) array of orthonormal eigenvectors of H
          (or at least close to eigenvectors)
        """
        # overlap matrix
        S = np.dot(v.conj().T,  np.dot(H, v))
        # eigf = (nwf_old, nwf_new)
        eigv, eigf = np.linalg.eigh(S)

        v_rot = np.dot(v, eigf)
        # need to rotate cg_dirs as well!
        self.cg_dirs = np.dot(self.cg_dirs, eigf)

        return v_rot




def orthonormalize(v):
    r"""Return orthonormalized set of vectors

    Return orthonormal set of vectors that spans the same space
    as the input vectors.

    parameters
    ----------
    v: array_like
      (n, m) array of m vectors in n-dimensional space
    """
    #orth = spla.orth(vecs.T)
    #orth /= np.linalg.norm(orth, axis=0)
    #return orth.T

    # vectors need to be columns
    Q, R = spla.qr(v, pivoting=False, mode='economic')

    # required to enforce positive signs of R's diagonal
    # without this, the signs of the orthonormalised vectors in Q are random
    # See https://mail.python.org/pipermail/scipy-user/2014-September/035990.html
    Q = Q * np.sign(np.diag(R))

    # Q contains orthonormalised vectors as columns
    return Q



class DiagMinimizer(object):
    """Performs minimization using exact diagonalisation

    This would be too slow for ab initio codes but is something we can afford
    in the iDEA world.

    Not yet fully implemented though (still missing analytic derivatives and a
    proper line search algorithm).
    """

    def __init__(self, pm, total_energy=None):
        """Initializes variables

        parameters
        ----------
        pm: object
          input parameters
        total_energy: callable
          function f(pm, waves) that returns total energy
        nstates: int
          how many states to retain.
          currently, this must equal the number of occupied states
          (for unoccupied states need to re-diagonalize)
        """
        self.pm = pm
        self.dx = pm.sys.deltax
        self.sqdx = np.sqrt(self.dx)
        self.dx2 = (self.dx)**2

        self.nstates = pm.sys.NE

        if total_energy is None:
            raise ValueError("Need to provide total_energy function that computes total energy from given set of single-particle wave functions.")
        else:
            self._total_energy = total_energy


    def total_energy(self, wfs):
        r"""Compute total energy for given wave function

        This method must be provided by the calling module
        and is initialized in the constructor.
        """
        return self._total_energy(self.pm, wfs/self.sqdx)


    def h_step(self, H0, H1):
        r"""Performs one minimisation step

        parameters
        ----------
        H0: array_like
          input Hamiltonian to be mixed (banded form)
        H1: array_like
          output Hamiltonian to be mixed (banded form)

        returns
        -------
        H: array_like
          mixed hamiltonian (banded form)
        """
        ## TODO: implement analytic derivative
        #dE_dtheta_0 = 2 * np.sum(self.braket(dirs, H, wfs).real)
        # For the moment, we simply fit the parabola through 3 points
        npt = 3
        lambdas = np.linspace(0, 1, npt)
        energies = np.zeros(npt)

        # self-consistent variant
        for i in range(npt):
            l = lambdas[i]
            Hl = (1-l) * H0 + l * H1
            enl, wfsl = spla.eig_banded(Hl, lower=True)
            energies[i] = self.total_energy(wfsl)
        
        # improve numerical stability
        energies -= energies[0]

        p = np.polyfit(lambdas, energies,2)
        a,b,c = p
        l = -b/ (2*a)

        # don't want step to get too large
        l = np.minimum(1,l)
        
        #import matplotlib.pyplot as plt
        #fig, axs = plt.subplots(1,1)
        #plt.plot(lambdas, energies)
        ##axs.set_ylim(np.min(total_energies), np.max(total_energies))
        #plt.show()

        Hl = (1-l) * H0 + l * H1

        return Hl
        
