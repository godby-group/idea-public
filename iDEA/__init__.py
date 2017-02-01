"""interacting Dynamic Electrons Approach (iDEA)

The iDEA code allows to propagate the time-dependent Schroedinger equation for
2-3 electrons in one-dimensional real space.
Compared to other models, such as the Anderson impurity model, this allows us
to treat exchange and correlation throughout the system and provides additional
flexibility in bridging the gap between model systems and ab initio
descriptions.
"""

# make Fortran libraries
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
import subprocess
# note: this could be made more clever to
#   automatically detect different environments
p = subprocess.Popen(["make"], cwd=dir_path)
