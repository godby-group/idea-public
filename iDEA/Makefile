# Select architecture (autodetect, if empty)
ARCH=

# Detect operating system
ifeq ($(OS),Windows_NT)
  OS_detected := Windows
else
  OS_detected := $(shell uname -s)
endif

# Set up environment variables for proper architecture
ifeq ($(ARCH),)
  ifeq ($(OS_detected),Windows)
    $(warning Warning: Windows currently untested!)
  else ifeq ($(OS_detected),Darwin)
    include arch/Darwin.mk 
  else ifeq ($(OS_detected),Linux)
    include arch/Linux.mk
  endif
else
  include arch/$(ARCH).mk
endif

# Handle versioned .so suffixes in python3
ifeq ($(PYTHON),python3)
    SO_EXT:=$(shell $(PY3CONFIG) --extension-suffix)
else
    SO_EXT:=.so
endif


###### Targets
default: parallel

parallel: RE_Utilities$(SO_EXT) construct_hamiltonian_coo2$(SO_EXT) construct_hamiltonian_coo3$(SO_EXT) construct_antisymmetry_coo2$(SO_EXT) construct_antisymmetry_coo3$(SO_EXT) construct_wavefunction2$(SO_EXT) construct_wavefunction3$(SO_EXT) 


%$(SO_EXT)::  %.f90
	$(F2PY) -c --fcompiler=$(FC) $(LFLAGS) -m $* $< --f90flags='$(FCFLAGS)' -lgomp --opt='$(OPTFLAGS)'

#%$(SO_EXT)::  %.f
#	$(F2PY) -c --fcompiler=$(FC) $(LFLAGS) -m $* $< --f77flags='$(FCFLAGS)' -lgomp --opt='$(OPTFLAGS)'

.PHONY: clean
clean:
	$(CLEAN)

