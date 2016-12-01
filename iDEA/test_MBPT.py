"""Tests for MBPT
""" 

import numpy as np
import numpy.testing as nt
import MBPT2
import input
import unittest

# decimal places for comparison of results
d = 10

class TestFFT(unittest.TestCase):
    """ Testing the Fourier transform in the (imaginary) time domain

    """ 

    def setUp(self):
        """ Sets up harmonic oscillator system """
        pm = input.Input()
        pm.sys.tau_npt = 101
        self.st = MBPT2.SpaceTimeGrid(pm)

    def test_fft_phase_1(self):
        """Test phase factors
        
        Sampling test function on shifted and unshifted grid.  Phase factors
        should bring Fourier components from both samples into agreement.
        """
        st = self.st
        grid_1 = st.tau_grid  # default grid dt/2, 3dt/2, ...
        grid_2 = st.tau_grid - st.tau_delta/2  # grid 0, dt, 2dt, ...

        def f(t):
            """Gaussian test function"""
            return np.array( np.exp(-0.5 * (t-1.5)**2), dtype=np.complex128)

        # Note: for functions that don't tend to zero at the edges of
        # the box, the phase shift can actually makes things worse
        #def f(t):
        #    """Exponential test function"""
        #    return np.array( np.exp(-0.05*t), dtype=np.complex128)

        f_w_1 = MBPT2.fft_t(f(grid_1),st, 't2f', phase_shift=True)
        #f_w_1 = MBPT2.fft_t(f(grid_1),st, 't2f', phase_shift=False)
        f_w_2 = np.fft.ifft(f(grid_2))* st.tau_npt * st.tau_delta

        # Note: f_w_1 and f_w_2 don't need to agree perfectly - the agreement should be better
        # than without phase shift
        nt.assert_array_almost_equal(f_w_1, f_w_2, decimal=8)

    def test_fft_phase_2(self):
        """Testing that Fourier transform with phases is reversible
        
        """
        st = self.st
        
        f_it = np.random.random(st.tau_npt) + 1J * np.random.random(st.tau_npt)
        f_iw = MBPT2.fft_t(f_it, st, dir='it2if', phase_shift=True)
        f_it_2 = MBPT2.fft_t(f_iw, st, dir='if2it', phase_shift=True)

        nt.assert_array_almost_equal(f_it, f_it_2, decimal=15)


    def test_fft_t_1(self):
        """Test that fft in real time gives expected results.
        
        This assumes a time grid that includes t=0.
        """
        st = self.st

        # Fourier transform of [1/st.deltax, ...] should be [1,1,...]
        f_t = np.zeros(st.tau_npt, dtype=complex)
        f_t[0] = 1/st.tau_delta
        f_w = MBPT2.fft_t(f_t,st,dir='t2f',phase_shift=False)
        nt.assert_array_almost_equal(f_w,np.ones(st.tau_npt), decimal=d)

        # Inverse Fourier transform of [1,1,...] should be [1/st.deltax, ...]
        f2_t = MBPT2.fft_t(f_w,st,dir='f2t',phase_shift=False)
        nt.assert_array_almost_equal(f_t,f2_t, decimal=d)


    def test_fft_it_1(self):
        """Test that fft in imaginary time gives expected results.
        
        This assumes an imaginary time grid that includes it=0.
        """
        st = self.st

        shift=False
        # Fourier transform of [1J/st.deltax, ...] should be [1,1,...]
        f_it = np.zeros(st.tau_npt, dtype=complex)
        f_it[0] = 1J/st.tau_delta
        f_iw = MBPT2.fft_t(f_it,st,dir='it2if',phase_shift=shift)
        nt.assert_array_almost_equal(f_iw,np.ones(st.tau_npt), decimal=d)

        # Inverse Fourier transform of [1,1,...] should be [1/st.deltax, ...]
        f2_it = MBPT2.fft_t(f_iw,st,dir='if2it',phase_shift=shift)
        nt.assert_array_almost_equal(f_it,f2_it, decimal=d)


#    def test_fft_it_2(self):
#        """Test that fft in imaginary time gives expected results.
#        
#        This assumes a shifhted imaginary time grid.
#        """
#        st = self.st
#
#        # Fourier transform of [1J/st.deltax, ...] should be [1,1,...]
#        f_iw = np.ones(st.tau_npt, dtype=complex)
#        f_it = MBPT2.fft_t(f_iw, st, dir='if2it', phase_shift=True)
#
#        #nt.assert_array_almost_equal(f_iw,np.ones(st.tau_npt), decimal=d)
#
#        print(st.tau_delta)
#        import matplotlib
#        matplotlib.use('Agg')
#        import matplotlib.pyplot as plt
#        plt.plot(st.tau_grid, f_it.real,'r.', label='real')
#        plt.plot(st.tau_grid, f_it.imag,'k.', label='imag')
#        plt.legend()
#        plt.savefig('test.png', dpi=300)


