# Tested on MacOS 10.10.5
# using gfortran 5.4 and MKL 11.3
# . /opt/intel/mkl/bin/mklvars.sh intel64
FC = gfortran
FCFLAGS = -m64 -I$(MKLROOT)/include -Wtabs
OPTFLAGS = -O3
LFLAGS  = -L$(MKLROOT)/lib -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread -liomp5 -ldl -lpthread -lm
CLEAN = rm -rf *.so *.dSYM

# specify 2 or 3 depending on your python version
PYTHON = 3
# f2py executable that is compatible with your python version
F2PY = f2py
# needed only for python 3
PY3CONFIG = python3-config
