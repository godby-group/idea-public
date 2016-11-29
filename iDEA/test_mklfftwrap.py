"""Tests for the mklfftwrap module

"""
import unittest
import numpy as np
import numpy.testing as nt
import mklfftwrap
import copy

class TestNumerics(unittest.TestCase):
    """ Testing against numpy.fft package

    Using random numbers as input.
    """

    def setUp(self):
        R=11
        T=401

        self.a = np.random.random_sample( (R,R,T)) \
          +1J* np.random.random_sample( (R,R,T))
        self.b = copy.deepcopy(self.a)

    def test_fft_1(self):
        """Testing fft_t against numpy's fft.fft"""
        a = mklfftwrap.fft_t(self.a)
        b = np.fft.fft(self.b, axis=-1)

        nt.assert_array_almost_equal(a,b, decimal=10)

    def test_ifft_1(self):
        """Testing ifft_t against numpy's fft.ifft"""
        a = mklfftwrap.ifft_t(self.a)
        b = np.fft.ifft(self.b, axis=-1)

        nt.assert_array_almost_equal(a/a.shape[-1],b, decimal=10)
