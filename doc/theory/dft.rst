Density Functional Theory (DFT)
===============================

A functional is a function of a function. This may sound like a scary
word, but you've definitely worked with them before! (e.g. an integral
is a functional :math:`\int f(x) dx`). The notation we're going to use
for functionals is square brackets, e.g. F[x] means F is a function of x
which is in turn a function of another variable, so F is a functional.

DFT is used to calculate the ground state of a system, and so is
independant of time. The most important function in DFT is the electron
density, :math:`n(\textbf{x})`. Once you've got the density, you are
able to calculate anything you want about the system (more on this
later). This vastly simplifies the problem, since the density is only a
function of 3 varibles, opposed to the 3N varibles of the full many body
wavefunction. The simplest formulations of DFT treat the
electron-electron interaction through a mean-field only approach, that
is reducing the many-body, interacting electron problem into many
non-interacting electrons moving in an effective potential.

Aside - Mean Field Theories
----------------------------

If you are familiar with the concept of a mean field theories, you might
like to think of DFT in this way but it isn't necessary to understand
the following. We'll include a quick reminder here of mean field theory
here in case you'd like to think of DFT in this way. One can think of
this as replacing operators by their expectation values, (or more
rigorously, writing an operator as its mean plus fluctuations about the
mean and then taking the 0th order term,
:math:`\hat{O} = \langle \hat{O} \rangle + \hat{\delta O} \approx \langle \hat{O} \rangle`.)
Clearly this means that the mean field has no fluctuations. This greatly
simplifies the problem since you don't need to keep track of each of the
individual particles and instead just one particle moving in the mean
field. (Add things about SCF)

Hohenberg-Kohn Theorems
-------------------------

DFT is built on two main theorems - the Hohenberg-Kohn theorems [Hohenberg1964]_ -
which we will state without proof here in the interests of brevity, but
we very much encourage you to look into their proofs - they're really
not so bad.

From this point on, we are going to specialise into 1 dimension to be
faithful to the iDEA code, but these ideas are easily generalised to 3D.

Theorem 1
..........

The external potential is a unique functional of the electron density
only (up to an additive constant). So the Hamiltonian, and therefore all
ground state properties, are determined solely by the electron density.

This is a very far reaching statement! Once we've got the electron
density, we have got everything we could want to know about the system.
It is important to recall how the density is related to the many-body
wave function:

.. math:: n(\textbf{x}) = N \int d{x}_2 \int d{x}_3 \ ... \int d{x}_N \mid \Psi ({x}, {x}_2, \ ... , {x}_N) \mid ^2

where the prefactor of N is included to account for the arbitrary
assignment of which of the electrons hasn't had its coordinate
integrated over.

Theorem 2
..........

The ground state energy may be obtained variationally, and the density
which minimises this energy is the exact ground state energy.

**NB** There is a nuance here, but a very important one!
During the proofs of these theorems, one assumes that the electrons are
in their ground states and so DFT is only valid for ground state
systems.

Now, taking these two theorems together prove that a universal
functional must exist, but sadly don't even give us a hint as to what it
should look like, (or even how to calculate the ground state energy).
Indeed, there are no known exact functionals for systems of more than
one electron! Also, you shouldn't get too excited by the electron
density being the central parameter, as ever things are more complicated
than they seem. Although *in principle* it is possible to
determine all properties of the the system from :math:`n(\textbf{x})`,
in practice it isn't that easy. The is reason is we often don't know
*how* to go from :math:`n(\textbf{x})` to the quantity
we're interested in finding and so have to revert back to the set of
:math:`N` wavefunctions.

At this point we pick up the *Kohn-Sham* formulation of DFT [Kohn1965]_, which is what
opened the door for so much progress in this field.

Kohn-Sham DFT
-------------

In the Kohn-Sham (KS) formulation of DFT, instead of considering the
full system of N interacting electrons, we instead look at a fictitious
system of N non-interacting electrons moving in an effective KS
potential, :math:`V_{KS}`. The single-particle KS orbitals are
constrained to give the same electron density as that of the real
system, so we can then, in theory at least, use theorem 1 to find out
anything we want about the system. This (KS density yielding the exact
density) is actually an assumption of the KS construction of the
fictitious system as no rigorous proofs for realistic systems. other
properties of the KS system are *not* the same as the
real system (e.g. the kinetic energy of the auxiliary system won't, in
general, be the same as that of the real system).

Recall that we are treating the electrons as spinless, so we constrain
the system such that there is only one electron per KS orbital which
gets around any possible problems arising from the Pauli exclusion
principal.

We're going to brush over a few of the formalities here, since, for
example, knowledge of how to use Lagrange multipliers to ensure particle
conservation doesn't give greater insight into the physics.

The goal from here is to solve the Schrödinger-like equation

.. math::  \bigg( - \frac{1}{2} \frac{d^2}{dx^2} + V_{KS}({x}) \bigg) \psi_i ({x}) = \varepsilon_i \psi_i ({x}),

where :math:`V_{KS}` is the Kohn-Sham potential that we'll discuss
shortly, and :math:`\varepsilon_i` are the single-partle eigenvalues. It
is worth emphasising that this equation is, in principal, exact.

The KS potential is given by

.. math::  V_{KS}({x}) = V_{ext}({x})+ V_H({x}) + V_x({x}) + V_c({x}),

where :math:`V_{ext}` is the external potential arising from the
electron's interaction with the nuclei, :math:`V_H` is the Hartree
potenial, :math:`V_x` is the exchange potential and :math:`V_c` the
correlation potential. The last two terms are often lumped together into
one exchange-correlation potential, :math:`V_{xc}`, which we shall use
fron now on.

Hartree potential
------------------

The easiest way to understand the origin of the Hartree potential is to
imagine freezing all the electrons in space, and then seeing what the
electrostatic potential is due to these electrons.

.. math::  V_H ({x}) = \int {n({x'})}u({\mid {x} - {x'}\mid}) d{x'},

where :math:`u` is the softened coulumb interaction implemented in iDEA.
If you examine definition of the Hartree potential you'll notice that it
includes self-intreaction, that is electron's interacting with other
parts of their *own* charge densities - don't worry, this
gets accounted for later. It's worth taking a moment here to reflect on
where we are so far. On the face of things, it might seem like we have
everything we need to solve this problem exactly. We've entirely
accounted for the Coulomb interaction between the electrons and the
nuclei and between the electrons themselves. So why do we need to bother
including :math:`V_{xc}`? The reason is that in defining :math:`V_H`, we
froze the electrons in place and got an *electrostatic*
potential, but of course the electrons in a real system will be moving,
and it is this movement that gives rise to the exchange-correlation
potential.

Exchange-Correlation potential
-------------------------------

The origin of the exchange part of the potential is due to the exchange
symmetry of the wavefunction of the system of identical particles (we'll
restrict our treatment to fermions here). When fermions get close to
each other they experience "Pauli repulsion", which causes the
expectation values between them to be larger. So when the electrons are
moving in the sample, they stay further away from each other than one
would naively expect. The correlation of the system is a bit harder to
put on explicit physical basis but it is a measure of how much the
motion of one electron affects that of another. :math:`V_{xc}` also
corrects for the self interaction in the Hartree potential.

The problem is that we don't know what the exchange-correlation
functional looks like for any system more complicated than the
homogenous electron gas (HEG), which is where KS DFT goes from being an
exact theory to an approximate one. We'll discuss one of these
approximations later.

DFT's strength lies in the fact that :math:`V_{xc}` is a relatively
small contribution to :math:`V_{KS}` so this term only being
approximately correct doesn't change the form of the KS potential too
drastically, which gives accruate KS oribitals and hence the electron
density given by

.. math:: n({x}) = \sum_i \mid\psi_i({x}) \mid ^2.

The alert reader may notice a problem here. We need the KS oribitals to
get the density by the above equation. To get the orbitals we need to
solve the Schrödinger-like equation, however, that requires knowledge of
the KS potential, which in turn depends on the electron density of the
system. So to solve this we put in a guess of the electron density
(often the density obtained from the non-interacting electron
approximation), then plug this into the Schrödinger-like equation for the
orbitals and then get the density from those. You then compare this new
density with the old one. If there has been a change, we plug this new
density in and try again. We keep this iteration going until we reach a
*self consistent solution*, or in practice that the
change from the old density to the new one is very small.

Of course this all assumes we know the form of the KS potential, but as
we mentioned earlier, no one knows the form of the exchange-correlation
functional which stops us doing this calculations exactly. One of the
most common approximations is to use the
*local density approximation* (LDA).

Local Density Approximation (LDA)
---------------------------------

In the LDA, the functional only depends on the place where we are
evaluating the density (hense the 'local' part of its name). The energy
functional is given by

.. math::  E_{xc}^{LDA}[n({x})] = \int \varepsilon_{xc}^{HEG}(n) \ n({x}) \ d{x},

where :math:`\varepsilon_{xc}^{HEG}(n)` is the exchange-correlation
energy per particle for the homogenous electron gas. Armed with this
functional, we can get :math:`V_{xc}^{LDA}` by using a functional
derivative, which is written as

.. math::  V_{xc}^{LDA} = \frac{\delta E_{xc}^{LDA}}{\delta n}.

Once, we have :math:`V_{xc}^{LDA}`, we can get the KS potential and go
through the process of finding a self consistent solution.

References
..........

.. [Hohenberg1964] "Inhomogeneous Electron Gas" P. Hohenberg and W. Kohn (1964) Phys. Rev. 136, B864

.. [Kohn1965] "Self-Consistent Equations Including Exchange and Correlation Effects" W. Kohn and L. J. Sham (1965) Phys. Rev. 140, A1133
