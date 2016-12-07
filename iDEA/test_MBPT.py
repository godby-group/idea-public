"""Tests for MBPT
""" 
from __future__ import division
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
        pm.mbpt.tau_npt = 161
        pm.mbpt.tau_max = 10
        self.pm = pm
        self.st = MBPT2.SpaceTimeGrid(pm)

    def test_fft_phase_1(self):
        """Test phase factors
        
        Sampling test function on shifted and unshifted grid.
        If the test function is continuous, then the phase factors should bring
        the Fourier components from both samples into (approximate) agreement.
        """
        st = self.st
        grid_s = st.tau_grid  # default shifted grid dt/2, 3dt/2, ...
        grid_z = st.tau_grid - st.tau_delta/2  # grid 0, dt, 2dt, ...

        def f(t):
            """Gaussian test function
            
            Note: This function is tuned such that it is very close to zero
            at the edges of the grid. Already a tiny jump at the edge of the 
            grid will cause significant deviations.
            """
            return np.array( np.exp(-0.20 * (t-1.345)**2), dtype=np.complex128)

        f_w_s = MBPT2.fft_t(f(grid_s),st, 't2f', phase_shift=True)
        f_w_z = np.fft.ifft(f(grid_z))* st.tau_npt * st.tau_delta

        # Note: f_w_1 and f_w_2 don't need to agree perfectly - the agreement should be better
        # than without phase shift
        nt.assert_array_almost_equal(f_w_s, f_w_z, decimal=6)



    def test_fft_phase_2(self):
        """Testing that Fourier transform with phases is reversible
        
        """
        st = self.st
        
        f_it = np.random.random(st.tau_npt) + 1J * np.random.random(st.tau_npt)
        f_iw = MBPT2.fft_t(f_it, st, dir='it2if', phase_shift=True)
        f_it_2 = MBPT2.fft_t(f_iw, st, dir='if2it', phase_shift=True)

        nt.assert_array_almost_equal(f_it, f_it_2, decimal=12)

    def test_fft_phase_3(self):
        """Test phase factors
        
        Sampling test function on shifted and unshifted grid.
        If the test function is continuous, then the phase factors should bring
        the Fourier components from both samples into (approximate) agreement.
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        st = self.st
        grid_z = st.tau_grid - st.tau_delta/2  # grid 0, dt, 2dt, ...
        grid_s = st.tau_grid  # default shifted grid dt/2, 3dt/2, ...

        def f(t):
            """Exponential test function"""
            return np.array( np.exp(-0.5*t), dtype=np.complex128)


        plt.plot(grid_z, f(grid_z).real, 'k.', label='unshifted')
        f_w_z = np.fft.ifft(f(grid_z))* st.tau_npt * st.tau_delta
        vmax = np.max(np.abs(f(grid_z)))
        plt.ylim([-vmax*1.2,vmax*1.2])
        #f_w_z = MBPT2.fft_t(f(grid_z),st, 't2f', phase_shift=False)

        plt.plot(grid_s, f(grid_s).real, 'r.', label='shifted')
        f_w_s = MBPT2.fft_t(f(grid_s),st, 't2f', phase_shift=True)

        nup = 8
        for shift in [True,False]:
            tau_npt = nup * st.tau_npt
            tau_delta = st.tau_delta /nup
            # note: this grid now *does* include tau=0 but not at index 0
            fpad_w = np.zeros(tau_npt, dtype=np.complex128)
            w_indices = np.round(np.fft.fftfreq(st.tau_npt) * st.tau_npt).astype(np.int)
            if shift:
                grid_up = 2*st.tau_max * np.fft.fftfreq(tau_npt) + st.tau_delta/2
                fpad_w[w_indices] = f_w_s * st.phase_forward
                if st.tau_npt % 2 == 0:
                    im = min(w_indices)
                    fpad_w[ -im ] = f_w_s[im] * st.phase_forward[im]
                clrs = ['r-','r--']
            else:
                grid_up = 2*st.tau_max * np.fft.fftfreq(tau_npt)
                fpad_w[w_indices] = f_w_z
                if st.tau_npt % 2 == 0:
                    im = min(w_indices)
                    fpad_w[ -im ] = f_w_z[im]
                clrs = ['k-','k--']
            f_t = np.fft.fft(fpad_w) / (tau_npt * tau_delta)

            srt = np.argsort(grid_up)
            plt.plot(grid_up[srt], f_t[srt].real,clrs[0], label='real')
            plt.plot(grid_up[srt], f_t[srt].imag,clrs[1], label='imag')

        plt.legend()
        plt.savefig('test_n{:+03d}.png'.format(st.tau_npt), dpi=300)
        plt.close()

        # Note: f_w_1 and f_w_2 don't need to agree perfectly - the agreement should be better
        # than without phase shift
        #nt.assert_array_almost_equal(f_w_s, f_w_z, decimal=6)



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
