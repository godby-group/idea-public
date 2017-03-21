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
        self.dx = pm.sys.deltax
        self.sqdx = np.sqrt(self.dx)
        self.dx2 = (self.dx)**2

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
          input wave functions(nwf, grid)
        H: array_like
          input Hamiltonian

        returns
        -------
        wfs: array_like
          updated wave functions
        """
        # internally work with dx=1
        wfs *= self.sqdx
        wfs = wfs[:self.nstates]

        energies = self.braket(wfs, H, wfs)

        steepest_dirs = self.steepest_dirs(H, wfs, energies)
        conjugate_dirs = self.conjugate_directions(steepest_dirs)

        theta_1 = np.pi/30
	# eqn (5.23)
        E_0 = self.total_energy(wfs)
        dE_dtheta_0 = 2 * np.sum(self.braket(conjugate_dirs, H, wfs).real)

        wfs_1 = wfs * np.cos(theta_1) + conjugate_dirs * np.sin(theta_1)
        wfs_1 = orthonormalize(wfs_1)
        E_1 = self.total_energy(wfs_1)

	#import matplotlib.pyplot as plt
	#x = np.linspace(0, np.pi/2, self.pm.sys.grid)
	#for i in range(2):
        #    plt.plot(x, conjugate_dirs[i], label='cg{}'.format(i))
        #    plt.plot(x, wfs[i], label=str(i))
        #    plt.plot(x, wfs_1[i], label=str(i))
	#plt.legend()
	#plt.show()

	# eqns (5.28-5.29)
	B_1 = 0.5 * dE_dtheta_0
	A_1 = (E_0 - E_1 + B_1) / (1 - np.cos(2*theta_1))

	# eqn (5.30)
	theta_opt = 0.5 * np.arctan(B_1/A_1)

        # solution from mathematica, apparently equivalent
	#theta_opt = - np.arctan( (A_1 + np.sqrt(A_1**2+B_1**2)) / B_1)

	#print(E_0)
	#print(E_1)
	#print(dE_dtheta_0)
	#print(theta_opt)

	#x = np.linspace(0, np.pi/2, 50)
	#y = A_1 * np.cos(2*x) + B_1 * np.sin(2*x)
	#import matplotlib.pyplot as plt
   	#plt.plot(x,y)
	#plt.show()

	wfs_new = wfs * np.cos(theta_opt) + conjugate_dirs * np.sin(theta_opt)
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
          lhs of braket
        O: array_like
          operator. defaults to identity matrix
        ket: array_like
          rhs of braket
        """
        if O is None:
	    if bra is None:
                return ket
            elif ket is None:
                return bra.conj().T
            else:
                return np.dot(bra.conj(), ket)
        else:
	    if bra is None:
                return np.dot(O, wfs.T).T
            elif ket is None:
                return np.dot(bra.conj(), O)
            else:
		O_ket = np.dot(O, ket.T).T
		return (bra.conj() * O_ket).sum(1)


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
        steepest = -(np.dot(H, wfs) - energies*wfs)

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


    def total_energy(self, wfs):
        r"""Compute total energy for given wave function

        This method must be provided by the calling module
        and is initialized in the constructor.
        """
        return self._total_energy(self.pm, wfs.T/self.sqdx)

    #def minimize(self):
    #    xopt = sopt.fmin_cg(E, R, psi, full_output=True)



def orthonormalize(vecs):
    r"""Return orthonormalized set of vectors

    Return orthonormal set of vectors that spans the same space
    as the input vectors.

    parameters
    ----------
    vecs: array_like
      (m,n) array of m vectors in n-dimensional space
    """
    #orth = spla.orth(vecs.T)
    #orth /= np.linalg.norm(orth, axis=0)
    #return orth.T

    v = vecs.T

    # vectors need to be columns
    Q, R = spla.qr(v, pivoting=False, mode='economic')

    # required to enforce positive signs of R's diagonal
    # without this, the signs of the orthonormalised vectors in Q are random
    # See https://mail.python.org/pipermail/scipy-user/2014-September/035990.html
    Q = Q * np.sign(np.diag(R))

    # Q contains orthonormalised vectors as columns
    return Q.T



