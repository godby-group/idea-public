"""Analytic continuation of the self-energy
"""
from __future__ import division
from __future__ import print_function
import numpy as np
import scipy.optimize as so
import copy as cp

class Polefit(object):
    r"""Fit function to sum of n poles.
     
    """

    def __init__(self, n_poles, fit_half_plane='upper', guess=None):
        """Constructs Polefit.

        parameters
        ----------
        n_poles: int
          number of poles of fitting function
        fit_half_plane: str
          half of the complex plane, where function is fitted and thus should
          be analytic (*free* of poles)

          - 'upper': fit f on upper half of complex plane
          - 'lower': fit f on lower half of complex plane

        guess: array_like
          provide guess for parameters of pole-function
        """
        self.n_poles = n_poles
        self.n_parameters = 2*n_poles + 1
        self.tries = 0

        n_parameters = self.n_parameters
        self.p  = np.empty((n_parameters), dtype=complex) 
        self.p0 = np.empty((n_parameters), dtype=complex) 

        ## following implementation in cp2k (for half_plane 1)
        #wmin = 0.0
        #wmax = 5.0
        #dw_re = (wmax - wmin) / (n_poles - 1)
        #dw_im = (wmax - wmin) / (n_poles)
        #self.p0[0] = 0.0
        #for i in range(1,n_poles+1):
        #    # pole strength a_j
        #    self.p0[2*i-1] = 1e-3
        #    # pole position b_j
        #    self.p0[2*i] = np.abs(wmin + (i-0.5) * dw_re) \
        #                   + 1J * (wmin + (i-1) * dw_im)


        self.fit_half_plane = fit_half_plane
        if guess is None:
            self.p0 = self.random_parameters()
        else:
            if len(guess) == n_parameters:
                self.p0 = guess
            else:
                raise ValueError("Guess does not contain {} parameters as required.".format(n_parameters))

    def random_parameters(self):
        r"""Returns random guess for parameters

        Parameters are random numbers with real parts in [-0.5,0.5] and
        imaginary parts in [-0.5,0.5].
        """
	guess = np.random.rand(self.n_parameters) - 0.5 \
                + 1J*(np.random.rand(self.n_parameters) - 0.5)
        return guess

    def cplx2float(self, cpl):
        r"""Store array of complex numbers in float array of twice the size"""
        #return np.array([cpl.real, cpl.imag]).flatten()
        return cpl.view(dtype=np.float)

    def float2cplx(self, flt):
        r"""Reconstruct array of complex numbers from float array twice the size"""
        return flt.view(dtype=np.complex)
        #n = int(len(flt) / 2)
        #cp = [flt[i] + 1J * flt[i + n] for i in range(n)]
        #return np.array(cp, dtype=complex)

    def fit_func(self, p, z):
        r"""Function to be fitted

        .. math:: 
            f(z) = a_0 + \sum_{j=1}^n \frac{a_j}{z-b_j}
                 = p_0 + \sum_{j=1}^n \frac{p_{2j-1}}{z-p_{2j}}
         
        """
        # testing alternative fct.
        return p[0] + p[2 - 1] / (z - p[2]) + p[4 - 1] / (z - p[4])

        #v = 0
        #for i in range(self.n_poles + 1):
        #    if i == 0:
        #        v += p[0]
        #    else:
        #        v += p[2 * i - 1] / (z - p[2 * i])

        #return v

    def df_dw(self, p, z):
        r"""Omega-derivative of model function

        .. math:: 
            f'(z) = -\sum_{j=1}^n \frac{a_j}{(z-b_j)^2}
                  = -\sum_{j=1}^n \left(\frac{p_{2j-1}}{(z-p_{2j})^2}

        May be used to compute quasiparticle weights via
        
        .. math::
            Z = 1/(1-f'(z)
        """
        v = 0
        for i in range(self.n_poles + 1):
            v -= p[2 * i - 1] / (z - p[2 * i])**2

        return v

    def qp_weight(self, en):
        r"""Compute quasiparticle weight

        parameters
        ----------
        en:  quasiparticle energy

        .. math::
            Z = 1/(1-f'(en))
        """
        w = 1.0 / (1.0 - self.df_dw(self.p,en))
        return w

    #def jacobian(self, p, x, y):
    #    """Computes Jacobian.

    #    returns
    #    -------
    #    jac_float
    #      1st index numbers parameters (2x number of complex parameters)
    #      2nd index numbers x (2x number of complex grid points)
    #      jac_float[0]  partial 
    #    """

    #    # first construct complex jacobian
    #    jac = np.empty((2*n_poles+1,len(x)),dtype=complex)
    #    for i in range(self.n_poles + 1):
    #        if i == 0:
    #            jac[i] = 1
    #        else:
    #            jac[2*i-1] = 1/(x-p[2*i])
    #            jac[2*i] = p[2*i-1] / (x-p[2*i])**2

    #    # for our particular function,
    #    # \pa f/\pa \Re a =   \pa f/\pa a
    #    # \pa f/\pa \Im a = i \pa f/\pa a
    #    jac_float = np.append(jac, 1J*jac, axis=0)
    #    # finally, we need to split jac_float into complex
    #    # and imaginary parts
    #    return np.append(jac_float.real,jac_float.imag, axis=0).T
    #    #return self.cplx2float(jac_float)


    def f(self, z):
        """Function using fitted parameters
        """
        return self.fit_func(self.p, z)


    def residuals(self, p, x, y):
        """Computes residual for scipy.optimize.leastsq.
        """
        res = self.fit_func(self.float2cplx(p), x) - y
        return self.cplx2float(res)

    def fit(self, x, y, n_tries=10, maxfev=10000):
        """Performs fit.
        """
        p0 = self.cplx2float(self.p0)

        # scipy.optimize.leastsq works best for parameters of O(1)
        # Thus we rescale y first and then put the scale factor back
        ys = cp.deepcopy(y)  # do not modify y (python passes by reference)
        yavg = np.average(y)
        ys -= yavg  # subtract (usually large) exchange component
        yscale = np.abs(ys).max()
        ys /= yscale 

        for _i in range(n_tries):
            self.tries += 1
            res = so.least_squares(self.residuals, p0, ftol=1e-8, args=(x, ys),
                    max_nfev=maxfev, method='lm')
            #p, cov = so.leastsq(self.residuals, p0, args=(x, y), maxfev=1000)

            if res.status in [1,2,3,4]:
                # solution was found
                break
            else:
                # solution was not found
                p0 = self.cplx2float(self.random_parameters())
                print("Polefit: Warning: Fit did not converge in {} iterations.\
                    Retrying with new initial guess...".format(res.nfev))
                continue

        p = self.float2cplx(res.x)
        # multiply all "a_j" parameters by scale and put avg back
        p[0] = p[0] * yscale + yavg  # this is the constant term a_0
        for j in range(self.n_poles):
            p[2*j+1] *= yscale      # these are the 'pole strengths'

        self.p = p
        self.res = res

    def random_fit(self, x, y, n_tries=10):
        """Fit with new randomized initial guess"""
        self.p0 = self.random_parameters()
        fit(x,y,n_tries)

    @property
    def poles(self):
        return np.array([self.p[2*i+2] for i in range(self.n_poles)]) 

 #   def get_fit_range(self, x, y, fit_half_plane):
 #       """returns appropriate fit range

 #       Fit range should include maxima of both real and imaginary part of y
 #       """
 #       for data in [np.abs(y.real), np.abs(y.imag]:
 #           max = np.max(np.abs(data))


    def check_poles(self):
        """Checks whether poles are in proper half of complex plane.

        Returns poles, which do not lie in correct half of the complex plane.
        """

        # When fitting on the upper imaginary axis, poles should lie in the 
        # lower complex half plane to allow for analytic continuation.
        if self.fit_half_plane == 'upper':
            return np.array([p for p in self.poles if p.imag >=0])
        elif self.fit_half_plane == 'lower':
            return np.array([p for p in self.poles if p.imag <=0])

