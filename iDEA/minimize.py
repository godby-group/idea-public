"""Direct minimisation of the Hamiltonian
"""
import numpy as np
#import scipy.optimize as sopt
import scipy.linalg as spla
import LDA


class CGMinimizer:
    """Performs conjugate gradient minimization

    Performs Pulay mixing with Kerker preconditioner,
    as described on pp 1071 of [Payne1992]_
    """

    def __init__(self, pm, total_energy=None, nstates=None, cg_restart=5, ndiag=20):
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
        ndiag: int
          wave functions are diagonalised every ndiag steps
          ndiag=0, diagonalisation is turned off.

        """
        self.pm = pm
        self.dx = pm.sys.deltax
        self.sqdx = np.sqrt(self.dx)
        self.dx2 = (self.dx)**2

        if nstates is None:
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

        self.ndiag = ndiag
        self.diag_counter = 0


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

        self.diag_counter += 1
        if self.diag_counter == self.ndiag:
            wfs = self.subspace_diagonalization(wfs,H)
            self.diag_counter = 0

        steepest_dirs = self.steepest_dirs(H, wfs)
        #print(np.linalg.norm(steepest_dirs, axis=0))
        conjugate_dirs = self.conjugate_directions(steepest_dirs, wfs)
        #print(np.linalg.norm(conjugate_dirs, axis=0))

        E_0 = self.total_energy(wfs)
        #dE_dtheta_0 = 2 * np.sum(self.braket(conjugate_dirs, H, wfs).real) / self.nstates
        dE_dtheta_0 = 2 * np.sum(self.braket(conjugate_dirs, H, wfs).real)

        if dE_dtheta_0 > 0:
            raise ValueError("First-order change along conjugate gradient direction is positive.")
            
        #import matplotlib.pyplot as plt
        #x = np.linspace(0, np.pi/2, self.pm.sys.grid)
        #for i in range(2):
        #    plt.plot(x, conjugate_dirs[i], label='cg{}'.format(i))
        #    plt.plot(x, wfs[i], label=str(i))
        #    plt.plot(x, wfs_1[i], label=str(i))
        #plt.legend()
        #plt.show()

        # line search
        mode = 'quadratic'

        if mode == 'trigonometric':
            # fitting sum of cosinus and sinus
            # note: payne et al. actually normalize the cg direction (?)
            s_1 = np.pi/30
            new_wfs = lambda s: wfs * np.cos(s) + conjugate_dirs * np.sin(s)


            # eqns (5.28-5.29)
            B_1 = 0.5 * dE_dtheta_0
            A_1 = lambda E_1, s: (E_0 - E_1 + B_1) / (1 - np.cos(2*s))
            s_min = lambda E_1, s: 0.5 * np.arctan(B_1 / A_1(E_1,s))

        elif mode == 'quadratic':
            # fitting simple parabola
            s_1 = 1.0 
            new_wfs = lambda s: wfs + conjugate_dirs * s

            b = lambda s: dE_dtheta_0 * s
            a = lambda E_1, s: (E_1 - E_0 - b(s)) / s**2
            s_min = lambda E_1, s: -0.5 * b(s) / (a(E_1,s) * s)


        while True:
            wfs_1 = new_wfs(s_1)
            wfs_1 = orthonormalize(wfs_1)
            E_1 = self.total_energy(wfs_1)

            break
            if E_1 < E_0:
                break
            else:
                s_1 /= 2.0

        #print("")
        # eqn (5.30)
        s_opt = s_min(E_1, s_1)
        # we don't want the step to get too large
        s_opt = np.minimum(1.5,s_opt) 

        #print("E_0: {}".format(E_0))
        #print("dE_0: {}".format(dE_dtheta_0))
        #print("E_1: {}".format(E_1))
        #print("step_1: {}".format(s_1))
        #print("step: {}".format(s_opt))
        #print("stepnorm: {}".format(s_opt * np.linalg.norm(conjugate_dirs, axis=0)))

        #if mode == 'trigonometric':
        #    x = np.linspace(0, np.pi/2, 50)
        #    y = A_1(E_1,s_opt) * np.cos(2*x) + B_1 * np.sin(2*x)
        #elif mode == 'quadratic':
        #    x = np.linspace(0, 1, 50)
        #    y = a(E_1,s_opt) * (x - s_opt)**2
        #import matplotlib.pyplot as plt
        #plt.plot(x,y)
        #plt.show()

        wfs_new = new_wfs(s_opt)
        wfs_new = orthonormalize(wfs_new)

        #theta = 0.8
        #converged = False
        #while not converged:
        #    wfs_new = wfs + theta * conjugate_dirs
        #    wfs_new = self.orthonormalize(wfs_new)
        #    break
        #    energies_new = self.expectation_values(?exp
        #    E = self.total_energy(self.pm, wfs_new)
        #    print(E)
        #    converged = True

        return wfs_new / self.sqdx

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


    def steepest_dirs(self, H, wfs):
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
          wave function array (grid, nwf)

        returns
        -------
        steepest_orth: array_like
          steepest descent directions (grid, nwf)
        """
        nwf = wfs.shape[-1]

        energies = self.braket(wfs, H, wfs)


        # steepest = (grid,nwf)
        # energies*wfs multiplies over last dimension of wfs
        steepest = -(np.dot(H, wfs) - energies*wfs)

        # eqn (5.12)
        # overlaps = (nwf, nwf)
        ## 1st index denotes wave function
        ## 2nd index denotes descent direction

        # note: varphi_i orthogonal to psi_i by construction thus no need to
        # restrict sum to j \neq i
        # note: not sure whether this is needed at all - it doesn't seem to do
        # anything (and it shouldn't, if the wfs are perfect eigenvectors of H)
        
        overlaps = np.dot(wfs.conj().T, steepest)
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
        gamma = np.sum(steepest_prods) / np.sum(self.steepest_prods)
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
        #cg_dirs = cg_dirs - np.sum(np.dot(cg_dirs.conj(), wfs.T)) * wfs
        self.cg_dirs = cg_dirs

        self.cg_counter += 1

        return cg_dirs


    def total_energy(self, wfs):
        r"""Compute total energy for given wave function

        This method must be provided by the calling module
        and is initialized in the constructor.
        """
        return self._total_energy(self.pm, wfs/self.sqdx)

    #def minimize(self):
    #    xopt = sopt.fmin_cg(E, R, psi, full_output=True)

     

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

