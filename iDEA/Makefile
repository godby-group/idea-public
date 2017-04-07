# Set up environment variables for proper architecture
ifeq ($(OS),Windows_NT)
    OS_detected := Windows
else
    OS_detected := $(shell uname -s)
endif

ifeq ($(OS_detected),Windows)
    $(warning Warning: Windows currently untested!)
else ifeq ($(OS_detected),Darwin)
    include arch/Darwin.mk 
else ifeq ($(OS_detected),Linux)
    include arch/Linux.mk
endif

###### Targets
default: parallel

parallel: mkl.so RE_Utilities.so construct_hamiltonian_coo2.so construct_hamiltonian_coo3.so construct_antisymmetry_coo2.so construct_antisymmetry_coo3.so construct_wavefunction2.so construct_wavefunction3.so 
serial: RE_Utilities

%.so::  %.f90
	f2py -c --fcompiler=$(FC) $(LFLAGS) -m $* $< --f90flags='$(FCFLAGS)' -lgomp --opt='$(OPTFLAGS)'

%.so::  %.f
	f2py -c --fcompiler=$(FC) $(LFLAGS) -m $* $< --f77flags='$(FCFLAGS)' -lgomp --opt='$(OPTFLAGS)'
#f2py -c --fcompiler=intelem -L${MKLROOT}/lib/intel64/ -lmkl_rt -m mkl mkl.f --f90flags='-openmp' -lgomp --opt='-fast'

.PHONY: clean
clean:
	$(CLEAN)

