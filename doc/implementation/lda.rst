LDA (Local Density Approximation)
=================================

The LDA is the most common approximation to the exchange-correlation (xc) functional in density functional theory (DFT). The LDA code implements 4 different functionals that we have developed [Entwistle2018]_- 3 were constructed from slab-like systems of 1, 2 and 3 electrons, and the other was constructed from the homogeneous electron gas (HEG), with our softened Coulomb interaction. These approximate xc potentials allow us to approximate the electron density of systems, for comparison with exact solutions. 

Calculating the ground-state
----------------------------

As an initial guess, the Kohn-Sham (KS) potential :math:`V_{\mathrm{KS}}` is approximated as the external potential :math:`V_{\mathrm{ext}}`. From this a set of non-interacting orbitals are computed through solving the KS (single-particle Schrödinger) equations:

.. math:: \{\phi_{i}, \varepsilon_{i}\},

and from these the density :math:`n(x)` is calculated. 

Using this density, an approximate xc potential :math:`V_{\mathrm{xc}}` is constructed using the chosen LDA functional. From this :math:`V_{\mathrm{KS}}` is refined, through a conjugate gradient method, Pulay mixing or linear mixing. For example, in linear mixing:

.. math:: V_{\mathrm{KS}}^{i+1}(x) = (1- \alpha)V^{i}_{\mathrm{KS}}(x) + \alpha V^{\mathrm{LDA}}_{\mathrm{KS}}(x),

where the Kohn-Sham potential at the current iteration :math:`V_{\mathrm{KS}}^{i}` and the Kohn-Sham potential constructed using the LDA :math:`V_{\mathrm{KS}}^{\mathrm{LDA}}` are mixed to generate the Kohn-Sham potential at the next iteration :math:`V_{\mathrm{KS}}^{i+1}`. 

These steps are repeated until we reach self-consistency, i.e. :math:`V_{\mathrm{KS}}(x)` and :math:`n(x)` are unchanging. 

Time-dependence
---------------

After an approximate ground-state :math:`V_{\mathrm{KS}}` is found, the perturbing potential is applied to the ground-state Hamiltonian, :math:`\hat{H} = \hat{H}_{0} + \delta V_{\mathrm{ext}}`. The system's evolution is calculated by propagating the ground-state KS orbitals through real time using the Crank-Nicholson method, and applying the LDA adiabatically (ALDA). 

References
----------

.. [Entwistle2018] M. T. Entwistle, M. Casula, and R. W. Godby, Phys. Rev. B 97, 235143 (2018).

.. [Kresse1996]	Kresse, G. & Furthmüller, J. Efficiency of ab-initio total energy calculations for metals and semiconductors using a plane-wave basis set. Computational Materials Science 6, 15–50 (1996). doi: 10.1016/0927-0256(96)00008-0

.. [Payne1992] Payne, M. P. Teter, D. C. Allan, T. A. Arias, and J. D. Joannopoulos, Rev. Mod. Phys. 64, 1045 (1992).

