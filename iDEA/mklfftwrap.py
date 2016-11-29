"""Interface to the FFT library of the Intel Math Kernel Library

Loads the dynamic library (.so/.dylib file) via the ctypes module.  
Provides specialized routine for 1d Fourier transform of nd arrays as required
by MBPT.
"""
import ctypes, ctypes.util
import numpy as np

# Constants from mkl_df_defines.h
DFTI_NUMBER_OF_TRANSFORMS = ctypes.c_int(7)
DFTI_PLACEMENT = ctypes.c_int(11)
DFTI_INPUT_DISTANCE = ctypes.c_int(14)
DFTI_OUTPUT_DISTANCE = ctypes.c_int(15)

DFTI_COMPLEX = ctypes.c_int(32)
DFTI_SINGLE = ctypes.c_int(35)
DFTI_DOUBLE = ctypes.c_int(36)

DFTI_NOT_INPLACE = ctypes.c_int(44)

#import os
#mkl = ctypes.cdll.LoadLibrary('{}/lib/libmkl_rt.dylib'.format(os.environ['MKLROOT']))

lib = ctypes.util.find_library('mkl_rt')
if lib is not None:
    mkl = ctypes.cdll.LoadLibrary(lib)
else:
    # In Linux, find_library does *not* search $LD_LIBRARY_PATH
    # This should be fixed in python >= 3.6
    # See https://bugs.python.org/issue9998
    try:
        mkl = ctypes.CDLL('libmkl_rt.so')
    except:
        raise RuntimeError("Library libmkl_rt not found in path")

#mkl.MKL_Set_Num_Threads(ctypes.c_int(1))

def desc_t(F):
    """
    Sets up descriptor for object F(r,r',t) or F(r,t)
    
    * transforms along last axis only
    * transform is not-in-place
    * implemented only for double precision complex numbers

    parameters
    ----------
    F: array

    returns
    -------
    desc_handle: FFT descriptor
    shape: proper shape for FFT
    """
    s = F.shape
    
    assert F.dtype == np.complex128
    assert F.flags['C_CONTIGUOUS']

    desc_handle = ctypes.c_void_p(0)
    if len(s) == 1:
        dim_fft = s[0]
        dim_not = 1
    elif len(s) >= 0:
        dim_fft = s[-1]
        dim_not = np.prod(s[0:-1])
    else:
        raise ValueError("F must be an array.")

    mkl.DftiCreateDescriptor(ctypes.byref(desc_handle), DFTI_DOUBLE, DFTI_COMPLEX, ctypes.c_long(1), ctypes.c_long(dim_fft) )
    mkl.DftiSetValue( desc_handle, DFTI_NUMBER_OF_TRANSFORMS, ctypes.c_long(dim_not) )
    if dim_not > 1:
        mkl.DftiSetValue( desc_handle, DFTI_INPUT_DISTANCE, ctypes.c_long(dim_fft) )
        mkl.DftiSetValue( desc_handle, DFTI_OUTPUT_DISTANCE, ctypes.c_long(dim_fft) )

    mkl.DftiSetValue( desc_handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE)
    mkl.DftiCommitDescriptor(desc_handle)

    shape = (dim_not, dim_fft)
    return desc_handle, shape


def fft_t(F):
    """
    Takes object F(r,r',t) and performs Fourier transform along last axis

    """
    shape = F.shape
    out = np.empty(shape, dtype=np.complex128)

    desc_handle, fft_shape = desc_t(F)

    F = F.reshape(fft_shape)
    mkl.DftiComputeForward(desc_handle, F.ctypes.data_as(ctypes.c_void_p), out.ctypes.data_as(ctypes.c_void_p) )
    out = out.reshape(shape)

    mkl.DftiFreeDescriptor(ctypes.byref(desc_handle))
    return out

def ifft_t(F):
    """
    Takes object F(r,r',t) and performs inverse Fourier transform along last axis

    """
    shape = F.shape
    out = np.empty(shape, dtype=np.complex128)

    desc_handle, fft_shape = desc_t(F)

    F = F.reshape(fft_shape)
    mkl.DftiComputeBackward(desc_handle, F.ctypes.data_as(ctypes.c_void_p), out.ctypes.data_as(ctypes.c_void_p) )
    out = out.reshape(shape)

    mkl.DftiFreeDescriptor(ctypes.byref(desc_handle))
    return out

