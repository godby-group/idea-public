# Tested on Ubuntu 16.04
# using ifort 17.0.1 and MKL 11.3
# Make sure to set Intel environment variables, like
# . /phys/sfw/intel/current/bin/compilervars.sh intel64
FC = intelem
FCFLAGS = -fast
OPTFLAGS = -fast
LFLAGS =  -L$(MKLROOT)/lib/intel64/ -lmkl_rt
CLEAN = rm -rf *.so

# specify 2 or 3 depending on your python version
PYTHON = 3
# f2py executable that is compatible with your python version
F2PY = f2py
PY3CONFIG = python3-config
