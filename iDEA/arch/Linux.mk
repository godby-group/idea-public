# Tested on Ubuntu 16.04
# using ifort 17.0.1 and MKL 11.3
# Make sure to set Intel environment variables, like
# . /phys/sfw/intel/current/bin/compilervars.sh intel64
FC = intelem
FCFLAGS = -fast
OPTFLAGS = -fast
LFLAGS =  -L$(MKLROOT)/lib/intel64/ -lmkl_rt
CLEAN = rm -rf *.so

F2PY = f2py
# build for python2 or python3
PYTHON = python3
PY3CONFIG = python3-config
