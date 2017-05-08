# TODO!
# 
# . /opt/intel/mkl/bin/mklvars.sh intel64
#FC = gfortran
#FCFLAGS = -m64 -I$(MKLROOT)/include -Wtabs
#LFLAGS  = -L$(MKLROOT)/lib -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread -liomp5 -ldl -lpthread -lm
CLEAN = del *.so 

PYTHON = 3   # specify 2 or 3 depending on your python version
F2PY = f2py  # f2py executable that is compatible with your python version
PY3CONFIG = python3-config  # needed only for python 3
