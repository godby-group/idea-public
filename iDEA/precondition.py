"""Preconditioners for self-consistent calculations.

Can be used in conjunction with mixing schemes (see mix.py).
"""
from __future__ import division
from __future__ import absolute_import

import numpy as np
import scipy.special as scsp

class StubPreconditioner(object):
    """Performs no preconditioning

    """

    def __init__(self, pm):
        """Initializes variables

        parameters
        ----------
        pm: object
          input parameters
        """

    def precondition(self, f, eigv, eigf):
        """Return preconditioned f"""
        return f



class KerkerPreconditioner(object):
    """Performs Kerker preconditioning

    Performs Kerker preconditioning,
    as described on p.34 of [Kresse1996]_

    """

    def __init__(self, pm):
        """Initializes variables

        parameters
        ----------
        pm: object
          input parameters
        """
        self.x_npt = pm.sys.grid
        self.x_delta = pm.sys.deltax

        # eqn (82)
        self.A = pm.lda.mix
        #kerker_length: float
        #  screening distance for Kerker preconditioning [a0]
        #  Default corresponds to :math:`2\pi/\lambda = 1.5\AA`

        self.q0 = 2*np.pi / pm.lda.kerker_length
        dq = 2*np.pi / (2 * pm.sys.xmax)
        q0_scaled = self.q0/ dq
        self.G_q = np.zeros((self.x_npt//2+1), dtype=np.float)
        for q in range(len(self.G_q)):
            self.G_q[q] = self.A * q**2 / (q**2 + q0_scaled**2)
        #self.G_r = np.set_diagonal(diagonal

        # one-dimensional Kerker mixing
        a = pm.sys.acon
        q = dq* np.array(list(range(self.x_npt//2+1)))
        aq = np.abs(a*q)
        Si, Ci = scsp.sici(aq)
        # verified that this agrees with Mathematica...
        v_k = -2*(np.cos(aq)*Ci + np.sin(aq)*(Si-np.pi/2))
        self.G_q_1d = self.A * 1/(1 + v_k * self.q0)

    def precondition(self, f, eigv, eigf):
        """Return preconditioned f"""
        f = np.fft.rfft(f, n=self.x_npt)
        f *= self.G_q
        f = np.fft.irfft(f, n=self.x_npt)
        return f



class RPAPreconditioner(object):
    """Performs preconditioning using RPA dielectric function

    The static dielectric function as a function of x and x'
    is computed in the Hartree approximation.

    """

    def __init__(self, pm):
        """Initializes variables

        parameters
        ----------
        pm: object
          input parameters
        """
        self.x_npt = pm.sys.grid
        self.x_delta = pm.sys.deltax
        self.NE = pm.sys.NE
        self.pm = pm

       # set up coulomb repulsion matrix v(i,j)
        tmp = np.empty((self.x_npt, self.x_npt), dtype=int)
        for i in range(self.x_npt):
            for j in range(self.x_npt):
                tmp[i,j] = np.abs(i - j)
        self.coulomb_repulsion = 1.0/(tmp * self.x_delta + pm.sys.acon)

    def precondition(self, r, eigv, eigf):
        r"""Preconditioning using RPA dielectric matrix

        .. math :: 
            
            \frac{\delta V(x)}{\delta \rho(x')} = DXC(x) \delta(x-x') + v(x-x')
        
        parameters
        ----------
        r: array_like
          array of residuals to be preconditioned
        eigv: array_like
          array of eigenvalues
        eigf: array_like
          array of eigenfunctions
        
        """
        dx = self.x_delta
        nx = self.x_npt
        v = self.coulomb_repulsion
        n = np.sum(np.abs(eigf[:self.NE])**2, axis=0)

        M = np.zeros((nx,nx))
        # note: this circular dependency only works in python 2.x
        # for python 3 let the constructur take the DXC function 
        # as a parameter (see e.g. mix.py for examples of this)
        from . import LDA
        np.fill_diagonal(M, LDA.DXC(self.pm, n)/dx)
        M += v

        chi = self.chi(eigv,eigf)
        # this is the correct recipe for *density* mixing.
        eps = np.eye(nx)/dx - np.dot(chi,M)*dx
        # for *potential* mixing use
        #eps = np.eye(nx)/dx - np.dot(M,chi)*dx

        epsinv = np.linalg.inv(eps)/dx**2
        r = np.dot(epsinv, r.T) * dx
        return r.T

    def chi(self,eigv, eigf):
        r"""Computes RPA polarizability

        The static, non-local polarisability (aka density-potential response) in
        the Hartree approximation (often called RPA):

        .. math ::
            \chi^0(x,x') = \sum_j^{'} \sum_k^{''} \phi_j(x)\phi_k^*(x)\phi_j^*(x')\phi_k(x') \frac{2}{\varepsilon_j-\varepsilon_k}

        where :math:`\sum^{'}` sums over occupied states and :math:`\sum^{''}` sums over empty states

        See also https://wiki.fysik.dtu.dk/gpaw/documentation/tddft/dielectric_response.html

        parameters
        ----------
        eigv: array_like
          array of eigenvalues
        eigf: array_like
          array of eigenfunctions

        returns
        -------
        epsilon: array_like
          dielectric matrix in real space
        """

        N = np.minimum(len(eigv), 10*self.NE)
        nx = self.x_npt

        chi = np.zeros((nx,nx))
        #for j in range(0,self.NE):
        #    for k in range(self.NE,N):
        #        for ix1 in range(self.x_npt):
        #            eps[ix1, :] += eigf[j,ix1]*np.conj(eigf[k,ix1])*np.conj(eigf[j,:])*eigf[k,:] *2.0/(eigv[j] - eigv[k])

        # eigenvalues should anyhow be real...
        eigv = eigv.real

        for j in range(0,self.NE):
            for k in range(self.NE,N):
                p1 = eigf[j] * np.conj(eigf[k])
                p2 = np.conj(p1)
                tmp = np.outer(p1,p2)
                chi += tmp.real * 2.0/(eigv[j] - eigv[k])
                #if j==self.NE-1 and k==self.NE:
                #    print("")
                #    print('{:.3e}'.format(eigv[j] - eigv[k]))

        #eps = np.dot(self.coulomb_repulsion,eps)*self.x_delta
        #eps = np.eye(nx)/self.x_delta - eps
        
        return chi
