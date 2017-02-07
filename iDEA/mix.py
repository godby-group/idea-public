"""Mixing schemes for self-consistent calculations
"""
import numpy as np

class PulayMixer:
    """Performs Pulay mixing

    Performs Pulay mixing with Kerker preconditioner,
    as described on p.34 of [Kresse1996]_

    """
    def __init__(self, pm, order, kerker_length=2.2):
        """Initializes variables

        parameters
        ----------
        order: int
          order of Pulay mixing (how many densities to keep in memory)
        pm: object
          input parameters
        kerker_length: float
          screening distance for Kerker preconditioning [a0]
          Default corresponds to :math:`2\pi/\lambda = 1.5\AA`

        """
        self.order = order
        self.step = 0
        self.x_npt = pm.sys.grid

        dtype = np.float
        self.res = np.zeros((order,self.x_npt), dtype=dtype)
        self.den_in = np.zeros((order,self.x_npt), dtype=dtype)

        self.den_delta = np.zeros((order-1,self.x_npt), dtype=dtype)
        self.res_delta = np.zeros((order-1,self.x_npt), dtype=dtype)

        # eqn (82)
        self.A = 0.8
        self.q0 = 2*np.pi / kerker_length
        dq = 2*np.pi / (2 * pm.sys.xmax)
        q0_scaled = self.q0 / dq
        self.G_q = np.zeros((self.x_npt/2+1), dtype=np.float)
        for q in range(len(self.G_q)):
            self.G_q[q] = self.A * q**2 / (q**2 + q0_scaled**2)
        #self.G_r = np.set_diagonal(diagonal


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

    def mix(self, den_in, den_out):
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
            den_in_new = den_in + self.precondition(self.res[m])
        else:
            ncoef = np.minimum(self.step, self.order)
            alpha_bar = self.compute_coefficients(m, ncoef)

            # eqn (92)
            den_in_new = den_in + self.precondition(self.res[m]) \
                + np.dot(alpha_bar, self.den_delta[:ncoef] + self.precondition(self.res_delta[:ncoef]))


        self.step = self.step + 1

        # this is a cheap fix for negative density values
        # but apparently that's what they do in CASTEP as well...
        return den_in_new.clip(min=0)

    def precondition(self, f):
        """Return preconditioned f"""
        #import matplotlib.pyplot as plt
        #x = np.linspace(-10,10,self.x_npt)
        #plt.plot(x,f)
        return f

        f = np.fft.rfft(f, n=self.x_npt)
        f = self.G_q * f
        f = np.fft.irfft(f, n=self.x_npt)
        #plt.plot(x,f)
        #plt.show()
        return f

