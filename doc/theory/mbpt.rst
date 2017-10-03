Many-Body Perturbation-Theory (MBPT)
------------------------------------

Many-body perturbation theory is a method used to compute electronic structure properties of systems in condensed matter.
The properties of the system can be determined by iteratively solving Hedin's equations. The GW approximation (GWA)
is a widely used approximation to the self-energy, which neglects complicated vertex corrections. Within the GWA the polarizability
of the system is approximated using the random-phase approximation (RPA). The :math:`GW` method can be used in many flavors:
one-shot (:math:`G_{0}W_{0}`), semi self-consistency (:math:`GW_0`) and full self-consistency (:math:`GW`). We use the space-time method to solve Hedin's equations.
We focus on the computation of the electron density using the GWA, which can be computed from the Green's function in
the imaginary-time domain. THe more technical details of the GW approximation are described `here <https://www.cmt.york.ac.uk/group_info/group/rwg3/Jack%20Wetherell%20first-year%20PhD%20report%202016.pdf>`_.
