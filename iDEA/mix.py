"""Mixing schemes for self-consistent calculations
"""
import numpy as np
import scipy.special as scsp



class PulayMixer:
    """Performs Pulay mixing

    Performs Pulay mixing with Kerker preconditioner,
    as described on p.34 of [Kresse1996]_

    """
    def __init__(self, pm, order, preconditioner='kerker'):
        """Initializes variables

        parameters
        ----------
        order: int
          order of Pulay mixing (how many densities to keep in memory)
        pm: object
          input parameters
        preconditioner: string
          May be Non, 'kerker' or 'full' 

        """
        self.order = order
        self.step = 0
        self.x_npt = pm.sys.grid
        self.x_delta = pm.sys.deltax
        self.NE = pm.sys.NE
	self.mixp = pm.lda.mix

        dtype = np.float
        self.res = np.zeros((order,self.x_npt), dtype=dtype)
        self.den_in = np.zeros((order,self.x_npt), dtype=dtype)

        self.den_delta = np.zeros((order-1,self.x_npt), dtype=dtype)
        self.res_delta = np.zeros((order-1,self.x_npt), dtype=dtype)

        if preconditioner == 'kerker':
            self.preconditioner = KerkerPreconditioner(pm)
        elif preconditioner == None:
            self.preconditioner = StubPreconditioner(pm)
        elif preconditioner == 'full':
            self.preconditioner = FullPreconditioner(pm)
        else:
            raise ValueError("Unknown preconditioner {}".format(preconditioner))


    def update_arrays(self, m, den_in, den_out):
        r"""Updates densities and residuals

        We need to store:
         * delta-quantities from i=1 up to m-1
         * den_in i=m-1, m
         * r i=m-1, m

        In order to get Pulay started, we do one Kerker-only step (step 0).

        Note: When self.step becomes larger than self.order,
        we overwrite data that is no longer needed.

        parameters
        ----------
        m: int
          array index for non-delta quantities
        den_in: array_like
          input density
        den_out: array_like
          output density
        """
        # eqn (88)
        self.den_in[m] = den_in
        if self.step > 0:
            self.den_delta[m-1] = self.den_in[m] - self.den_in[m-1]

        self.res[m] = den_out - den_in
        if self.step > 0:
            self.res_delta[m-1] = self.res[m] - self.res[m-1]

    def compute_coefficients(self,m, ncoef):
        r"""Computes mixing coefficients
        
        See [Kresse1996]_ equations (87) - (90)

        .. math ::

            A_{ij} = \langle R[\rho^j_{in}] | R[\rho_{in}^i \rangle \\
            \bar{A}_{ij} = \langle \Delta R^j | \Delta R^i \rangle

        See [Kresse1996]_ equation (92)
        
        parameters
        ----------
        m: int
          array index for non-delta quantities
        ncoef: int
          number of coefficients to compute
        """

        # we return rhoin_m+1, which needs
        # * den_in m
        # * r m
        # * delta_dens i=1 up to m-1
        # * delta_rs i=1 up to m-1
        # * alpha_bars i=1 up to m-1

        # * delta_den m-1 needs den m-1, den m 
        # * delta_r m-1 needs r m-1, r m


        # eqns (90,91)
        # Note: In principle, one should multiply by dx for each np.dot operation.
        #       However, in the end these must cancel out
        # overlaps / dx
        overlaps = np.dot(self.res_delta[:ncoef], self.res[m])
        # A_bar / dx
        A_bar = np.dot(self.res_delta[:ncoef], self.res_delta[:ncoef].T)
        # A_bar_inv / dx * dx**2 = A_bar_inv * dx
        A_bar_inv = np.linalg.inv(A_bar)
        # alpha_bar * dx / dx / dx = alpha_bar / dx
        alpha_bar = -np.dot(A_bar_inv.T,overlaps)

        return alpha_bar

    def mix(self, den_in, den_out, eigv=None, eigf=None):
        r"""Compute mix of densities

        Computes new input density rho_in^{m+1}, where the index m corresponds
        to the index m used in [Kresse1996]_ on pp 33-34.
        
        parameters
        ----------
        den_in: array_like
          input density
        den_out: array_like
          output density
        """

        m = self.step % self.order
        self.update_arrays(m, den_in, den_out)

        # for the first step, we simply do preconditioning
        if self.step == 0:
            den_in_new = den_in + self.precondition(self.res[m], eigv, eigf)
        else:
            ncoef = np.minimum(self.step, self.order)
            alpha_bar = self.compute_coefficients(m, ncoef)

            # eqn (92)
            den_in_new = den_in + self.precondition(self.res[m], eigv, eigf) \
                + np.dot(alpha_bar, self.den_delta[:ncoef] + self.precondition(self.res_delta[:ncoef], eigv, eigf))


        self.step = self.step + 1

        # this is a cheap fix for negative density values
        # but apparently that's what they do in CASTEP as well...
        return den_in_new.clip(min=0)

    def precondition(self, f, eigv, eigf):
        """Return preconditioned f"""
        return self.mixp * self.preconditioner.precondition(f, eigv, eigf)


class StubPreconditioner:
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



class KerkerPreconditioner:
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
        q0_scaled = self.q0 / dq
        self.G_q = np.zeros((self.x_npt//2+1), dtype=np.float)
        for q in range(len(self.G_q)):
            self.G_q[q] = self.A * q**2 / (q**2 + q0_scaled**2)
        #self.G_r = np.set_diagonal(diagonal

        # one-dimensional Kerker mixing
        a = pm.sys.acon
        q = dq* np.array(range(self.x_npt//2+1))
        aq = np.abs(a*q)
        Si, Ci = scsp.sici(aq)
        # verified that this agrees with Mathematica...
        v_k = -2*(np.cos(aq)*Ci + np.sin(aq)*(Si-np.pi/2))
        self.G_q_1d = self.A * 1/(1 + v_k * self.q0)

    def precondition(self, f, eigv, eigf):
        """Return preconditioned f"""
        #import matplotlib.pyplot as plt
        #x = np.linspace(-10,10,self.x_npt)
        #plt.plot(x,f)

        f = np.fft.rfft(f, n=self.x_npt)
        f *= self.G_q
        f = np.fft.irfft(f, n=self.x_npt)
        #plt.plot(x,f)
        #plt.show()
        return f



class FullPreconditioner:
    """Performs preconditioning using full dielectric function

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

       # set up coulomb repulsion matrix v(i,j)
        tmp = np.empty((self.x_npt, self.x_npt), dtype=int)
        for i in range(self.x_npt):
            for j in range(self.x_npt):
                tmp[i,j] = np.abs(i - j)
        self.coulomb_repulsion = 1.0/(tmp * self.x_delta + pm.sys.acon)

    def precondition(self, r, eigv, eigf):
        """Preconditioning using full dielectric matrix
        
        parameters
        ----------
        r: array_like
          array of residuals to be preconditioned
        eigv: array_like
          array of eigenvalues
        eigf: array_like
          array of eigenfunctions
        
        """
        #import matplotlib.pyplot as plt
        #x = np.linspace(-10,10,self.x_npt)
        #plt.plot(x,f)
        dx = self.x_delta
        nx = self.x_npt
        v = self.coulomb_repulsion

        chi = self.chi(eigv,eigf)
        # this is the correct recipe for *density* mixing.
        eps = np.eye(nx)/dx - np.dot(chi,v)*dx
        # for *potential* mixing use
        #eps = np.eye(nx)/dx - np.dot(v,chi)*dx

        epsinv = np.linalg.inv(eps)/dx**2 
        r = np.dot(epsinv, r.T) * dx
        return r.T

    def chi(self,eigv, eigf):
        r"""Computes RPA polarizability

        The static, non-local polarisability (or density-potential
        response) in the Hartree approximation (often called RPA)
        is computed as

        .. math ::
            \chi^0(x,x') = \sum_j^{'} \sum_k^{''} \phi_j(x)\phi_k^*(x)\phi_j^*(x')\phi_k(x') \frac{2}{\varepsilon_j-\varepsilon_k}

        where :math:`\sum^'` sums over occupied states and :math:`\sum^{''}` sums over empty states

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

        #print(np.sum(eigf[1]**2)*self.x_delta)
   	#print("max {}".format(np.max(eps)))
   	#print("min {}".format(np.min(eps)))
   	#print("max {}".format(np.max(eps)))
   	#print("min {}".format(np.min(eps)))


        #eps = np.dot(self.coulomb_repulsion,eps)*self.x_delta
        #eps = np.eye(nx)/self.x_delta - eps
	
        #import matplotlib.pyplot as plt
        #x = np.linspace(-10,10,self.x_npt)
        #for i in [2,3]:
        #    plt.plot(x,np.abs(eigf[i]), label=i)
        #plt.legend()
        #plt.show()
	#print(np.diagonal(eps))
        return chi



