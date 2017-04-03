# Tested on MacOS 10.10.5
# using gfortran 5.4 and MKL 11.3
# . /opt/intel/mkl/bin/mklvars.sh intel64
FC = gfortran
FCFLAGS = -m64 -I$(MKLROOT)/include -Wtabs
LFLAGS  = -L$(MKLROOT)/lib -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread -liomp5 -ldl -lpthread -lm
CLEAN = rm -rf *.so *.dSYM
