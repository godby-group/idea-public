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

# work around versioned .so library suffixes in python3
SO_EXT2:=.so
SO_EXT3:=$(shell python3-config --extension-suffix)

###### Targets
default: python2

python2: F2PY=f2py
python2: RE_Utilities$(SO_EXT2) construct_hamiltonian_coo2$(SO_EXT2) construct_hamiltonian_coo3$(SO_EXT2) construct_antisymmetry_coo2$(SO_EXT2) construct_antisymmetry_coo3$(SO_EXT2) construct_wavefunction2$(SO_EXT2) construct_wavefunction3$(SO_EXT2) 

python3: F2PY=f2py3
python3: RE_Utilities$(SO_EXT3) construct_hamiltonian_coo2$(SO_EXT3) construct_hamiltonian_coo3$(SO_EXT3) construct_antisymmetry_coo2$(SO_EXT3) construct_antisymmetry_coo3$(SO_EXT3) construct_wavefunction2$(SO_EXT3) construct_wavefunction3$(SO_EXT3) 


# this rule needs to come first as SO_EXT3 is typically longer
%$(SO_EXT3)::  %.f90
	$(F2PY) -c --fcompiler=$(FC) $(LFLAGS) -m $* $< --f90flags='$(FCFLAGS)' -lgomp --opt='$(OPTFLAGS)'

%$(SO_EXT2)::  %.f90
	$(F2PY) -c --fcompiler=$(FC) $(LFLAGS) -m $* $< --f90flags='$(FCFLAGS)' -lgomp --opt='$(OPTFLAGS)'

#%$(SO_EXT2)::  %.f
#	$(F2PY) -c --fcompiler=$(FC) $(LFLAGS) -m $* $< --f77flags='$(FCFLAGS)' -lgomp --opt='$(OPTFLAGS)'
#f2py -c --fcompiler=intelem -L${MKLROOT}/lib/intel64/ -lmkl_rt -m mkl mkl.f --f90flags='-openmp' -lgomp --opt='-fast'

.PHONY: clean
clean:
	$(CLEAN)

