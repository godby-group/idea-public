NON (non-interacting)
=====================

The NON code solves the single-particle time-independent Schrödinger equation (TISE) for one-dimensional finite systems of non-interacting electrons. A perturbing potential in then applied to the ground-state system and its evolution is calculated through solving the single-particle time-dependent Schrödinger equation (TDSE).

Calculating the ground-state
----------------------------

The Hamiltonian of a system of non-interacting electrons subject to a specified external potential is constructed, and from this a set of non-interacting orbitals are computed through solving the single-particle TISE:

.. math:: \{\phi_{i}, \varepsilon_{i}\}.

From this, the ground-state density is calculated:

.. math:: n(x) = \sum_{j \in \text{occ}} | \phi_{j} | ^{2}.

Time-dependence
---------------

After the ground-state is found, the perturbing potential is applied to the ground-state Hamiltonian, :math:`\hat{H} = \hat{H}_{0} + \delta V_{\mathrm{ext}}`. The system's evolution is calculated by propagating the ground-state orbitals through real time using the Crank-Nicholson method.
