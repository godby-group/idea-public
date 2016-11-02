# Set up environment variables for proper architecture
#include make_rwgu.sys
include make_macbook.sys

###### Targets
default: all

all: mkl RE_Utilities create_hamiltonian_coo

%::  %.f90
	f2py -c --fcompiler=$(FC) $(LFLAGS) -m $@ $< --f90flags='$(FCFLAGS)' -lgomp --opt='$(OPTFLAGS)'

%::  %.f
	f2py -c --fcompiler=$(FC) $(LFLAGS) -m $@ $< --f77flags='$(FCFLAGS)' -lgomp --opt='$(OPTFLAGS)'


#f2py -c --fcompiler=intelem -L${MKLROOT}/lib/intel64/ -lmkl_rt -m mkl mkl.f --f90flags='-openmp' -lgomp --opt='-fast'
