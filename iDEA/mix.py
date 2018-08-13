"""Mixing schemes for self-consistent calculations
"""
from __future__ import division
from __future__ import absolute_import

import numpy as np
import scipy.special as scsp

from . import precondition

class PulayMixer(object):
    """Performs Pulay mixing

    Can perform Pulay mixing with Kerker preconditioning as described on p.34 of [Kresse1996]_
    Can also be combined with other preconditioners (see precondition.py).
    """
    def __init__(self, pm, order, preconditioner=None):
        """Initializes variables

        parameters
        ----------
        order: int
          order of Pulay mixing (how many densities to keep in memory)
        pm: object
          input parameters
        preconditioner: string
          May be None, 'kerker' or 'rpa' 

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
            self.preconditioner = precondition.KerkerPreconditioner(pm)
        elif preconditioner == None:
            self.preconditioner = precondition.StubPreconditioner(pm)
        elif preconditioner == 'rpa':
            self.preconditioner = precondition.RPAPreconditioner(pm)
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
