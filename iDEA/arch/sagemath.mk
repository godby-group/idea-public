# Tested 2017-04-26 on compute0-us.sagemath.com
# Using Python 3.5.3 from Anaconda
MKLROOT = /projects/anaconda3
FC = gfortran
FCFLAGS = -m64 -I$(MKLROOT)/include -Wtabs
OPTFLAGS = $(FCFLAGS)
LFLAGS  = -L$(MKLROOT)/lib -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread -liomp5 -ldl -lpthread -lm
CLEAN = rm -rf *.so

F2PY = /projects/anaconda3/bin/f2py
# build for python2 or python3
PYTHON = python3
PY3CONFIG = /projects/anaconda3/bin/python3-config
