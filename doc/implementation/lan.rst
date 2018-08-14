LAN (Landauer)
==============

The LAN code implements the Landauer formulation, in which the electrons are non-interacting, to provide an approximate time-dependent electron density.

Ground-state
------------

A ground-state Kohn-Sham (KS) potential is provided - this can be the exact one that has been calculated through reverse-engineering, or an approximation e.g. LDA, HF etc. The Hamiltonian is constructed, and from this the KS equations are solved to calculate a set of non-interacting orbitals:

.. math:: \{\phi_{i}, \varepsilon_{i}\},

and from these the density :math:`n(x)` is calculated.

Time-dependence
--------------- 

After an approximate (or exact) ground-state electron density has been found, the perturbing potential is applied to the ground-state Hamiltonian, :math:`\hat{H} = \hat{H}_{0} + \delta V_{\mathrm{ext}}`. The time-dependent KS potential is assumed to be equal to the ground-state KS potential at all times, i.e. :math:`V_{\mathrm{KS}}(x,t) = V_{\mathrm{KS}}(x,0)`. The system's evolution is calculated by propagating the ground-state KS orbitals through real time using the Crank-Nicholson method.
