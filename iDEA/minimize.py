"""Direct minimisation of the Hamiltonian
"""
import numpy as np
import scipy.optimize as sopt
import scipy.linalg as spla
import LDA



class CGMinimizer:
    """Performs conjugate gradient minimization

    Performs Pulay mixing with Kerker preconditioner,
    as described on pp 1071 of [Payne1992]_

    """

    def __init__(self, pm, total_energy=None, nstates=None):
        """Initializes variables

        parameters
        ----------
        pm: object
          input parameters
        H: callable
          function returning Hamiltonian

        """
        self.pm = pm
        self.cg_dirs = np.zeros((pm.sys.NE, pm.sys.grid))
        self.steepest_prods = np.ones((pm.sys.NE))
        if total_energy is not None:
            self._total_energy = total_energy

        if nstates is None:
            self.nstates = pm.sys.NE
        else:
            self.nstates = nstates


    def gradient_step(self, wfs, H):
        r"""Performs one cg step

        After each step, the Hamiltonian should be recomputed using the updated
        wave functions.
        
        Note that we currently don't enforce the wave functions to remain
        eigenfunctions of the Hamiltonian. This should not matter for the total
        energy but means we need to performa a diagonalisation at the very end.

        parameters
        ----------
        wfs: array_like
          input wave functions
        H: array_like
          input Hamiltonian

        returns
        -------
        wfs: array_like
          updated wave functions
        """
        wfs = wfs[:self.nstates]
        energies = self.expectation_values(wfs, H)
        print(energies)
        steepest_dirs = self.steepest_dirs(H, wfs, energies)

        conjugate_dirs = self.conjugate_directions(steepest_dirs)

        theta = 0.8
        converged = False
        while not converged:
            wfs_new = wfs + theta * conjugate_dirs
            wfs_new = self.orthonormalize(wfs_new)
            #E = self.total_energy(wfs_new)
            converged = True

        return wfs_new

    def expectation_values(self, wfs, O):
        r"""Compute expectation values wrt operator O

        .. math:
            \lambda_i^m = \langle \psi_i^m | O | \psi_i^m \rangle

        See eq (5.11) in [Payne1992]_
        """
        O_psi = np.dot(O, wfs.T).T
        psi_O_psi = (wfs.conj()*O_psi).sum(1)
        return psi_O_psi

    def steepest_dirs(self, H, wavefunctions, energies):
        r"""Compute steepest descent directions

        Compute steepest descent directions and project out components pointing
        along other orbitals.

        .. math:
            \zeta^{'m}_i = -(H-\lambda_i^m)\psi_i^m - \sum_{j\neq i} \langle \psi_j|\zeta_i^m\rangle \psi_j

        See eqns (5.10), (5.12) in [Payne1992]_

        parameters
        ----------
        H: array_like
          Hamiltonian matrix (grid,grid)
        wavefunctions: array_like
          wave function array (nwf, grid)
        energies: array_like
          energies of wave functions (grid)
        """
        nwf=len(wavefunctions)
        wfs = wavefunctions.T # more convenient here

        # steepest = (grid,nwf)
        # energies*wfs multiplies over last dimension of wfs
        steepest = - np.dot(H, wfs) - energies*wfs

        # eqn (5.12)
        # overlaps = (nwf, nwf)
        # 1st index denotes wave function
        # 2nd index denotes descent direction
        overlaps = np.ma.array(np.dot(wfs.conj().T, steepest), mask=False)
        for i in range(nwf):
            overlaps.mask[i,i] = True # exclude index i in summation

        # need to contract over first index (wave functions)
        # steepest_orth = (grid, nwf)
        steepest_orth = steepest - np.ma.dot(wfs, overlaps)
        steepest_orth = np.ma.getdata(steepest_orth)

        return steepest_orth.T

    def conjugate_directions(self, steepest_dirs):
        r"""Compute conjugate gradient descent for one state

        Updates internal arrays accordingly

        .. math:
            d^m = g^m + \gamma^m d^{m-1}
            \gamma^m = \frac{g^m\cdot g^m}{g^{m-1}\cdot g^{m-1}}

        See eqns (5.8-9) in [Payne1992]_

        parameters
        ----------
        steepest_dirs: array_like
          steepest-descent directions
        """
        steepest_prods = np.linalg.norm(steepest_dirs)**2
        gamma = steepest_prods / self.steepest_prods
        self.steepest_prods = steepest_prods

        cg_dirs = steepest_dirs + np.dot(gamma, self.cg_dirs)
        self.cg_dirs = cg_dirs

        return cg_dirs

    def orthonormalize(self, vecs):
        r"""Return orthonormalized set of vectors

        Return orthonormal set of vectors that spans the same space
        as the input vectors.

        parameters
        ----------
        vecs: array_like
          (m,n) array of m vectors in n-dimensional space
        """
        orth = spla.orth(vecs.T)
        orth /= np.linalg.norm(orth, axis=0)


        return orth.T


    def total_energy(self, wfs):
        r"""Compute total energy for given wave function

        This method must be provided by the calling module
        and is initialized in the constructor.
        """
        return self._total_energy(wfs)

    #def minimize(self):
    #    xopt = sopt.fmin_cg(E, R, psi, full_output=True)

