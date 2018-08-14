Many-electron Quantum Mechanics
===============================

Motivation
----------

The theory of interacting electrons is extremely rich and complex. This, along
with recent advances in computing capabilities, means that more than 10,000
papers are published on electronic structure theory each year. This research is
yielding novel understanding in a vast range of fields including physics,
chemistry and materials science.

This is because, on the atomic level, electrons are the *glue* of matter and if
we understand how electrons move in their environment *and* how they interact
with each other, we can predict the electronic, optical and mechanical
properties of materials.


Notation
---------

We are going to use `Hartree atomic units <https://en.wikipedia.org/wiki/Atomic_units>`_ where
:math:`e = \hbar = m_e = 4 \pi \varepsilon_0 = 1` which saves a lot of
clutter! This means the standard unit of length is the Bohr radius
:math:`a_0 = 5.29 \times 10^{-11} \mathrm{m}`, and the unit of energy is
the Hartree :math:`E_H = 2\mathrm{Ry} = 27.2 \mathrm{eV}`. Also be aware that
capital :math:`\Psi` refers to a many-body wave function whereas lower
case :math:`\psi` refers to single particle wave functions.

Below, we keep everything in 3 dimensions to keep things general, but be aware
that iDEA works in 1D only.

Schrödinger equation
---------------------

The name of the game is to solve the Schrödinger equation for both the
ground state system, :math:`\hat{H} \Psi = E \Psi` and for the time
dependent one, :math:`\hat{H} \Psi = i \frac{\partial \Psi}{\partial t}`
(we'll consider the time dependent case later).

In an ideal world, we would solve the many-body Schrödinger equation
exactly and then armed with these wavefunctions, we'd have complete
knowledge of the system and be able to make precise predictions.
Unfortunately, however, the problem is much too hard to solve in the
way. Let's have a look at the Hamiltonian to see why.

.. math::  \hat{H} = - \frac{1}{2} \sum_i \nabla_i^2 + V_{ext}(\textbf{r}_i) + V_{ee}(\mid \textbf{r}_i - \textbf{r}_j \mid ),

where we have used the Born-Oppenheimer approximation (treating the
nuclei as so massive that they do not move on the timescale of electron
motion), and the fact that the nuclei-nuclei interacting is constant and
so we simply shift our zero of energy to absorb that term.
:math:`V_{ext}` is the external potential in which the electrons move,
in the case of electrons moving in a potential set up by the nuclei,
:math:`V_{ext} = -\sum_{i,I} \frac{Z_I}{\mid \textbf{r}_i - \textbf{R}_I\mid }`.
:math:`V_{ee}` is the electron-electron interaction which in 3
dimensions takes the form
:math:`V_{ee} =\frac{1}{2} \sum_{i \neq j} \frac{1}{\mid \textbf{r$_i$} - \textbf{r$_j$} \mid}`,
but of course has a different 1D form which is implemented in the iDEA code.

The difficulty arises with the third term, the electron-electron
interaction. This term means that the Schrödinger equation isn't
seperable so the wavefunction isn't simply a suitable anti-symmetric
product of single particle wavefunctions. We now explore several
approaches to solve this problem that are used by the iDEA code.

Exact solution
--------------

We are going to pretend from here that our electrons are spinless, so we
can ignore any spin-orbit effects. This also has more profound impacts
on the universe we're considering. In a spinful universe, only electrons
of equal spins would feel the exchange interaction, but in our spinless
universe all electrons feel it. This means we get to see the effects of
:math:`V_{xc}` for systems with a smaller number of electrons so we get
to maximise the amount of physics we can highlight for a given effort
(on both your part and mine!).

As we mentioned earlier, the exact problem is, normally, much too hard
to solve; the difficulty growing exponentially with the number of
electrons in the system. However, for systems with 2 or 3 electrons, we *are*
able to solve the problem exactly. In fact, this is one of the key concepts in
iDEA - solving simple systems exactly to allow us to compare, and possibly
improve, the approximate solutions. Armed with these improvements, we are in a
better position to tackle the larger systems with many electrons.

Given we can't solve the many-electron system exactly in general, we
need to have a look at the various different approaches of tackling the
problem in more complicated systems.


Complete neglect of interaction
---------------------------------

This is by far the simplest approach - you just completely ignore the
Coulomb interaction between the electrons. Clearly this massively
simplifies the problem, giving a separable Hamiltonian for a start! This
means that we are able to solve the Schrödinger equation by the method
of separation of variables, solving it for each electron individually.
Then the total wavefunction is just a product of these single particle
wave functions, in a suitable anti symmetric arrangement. One often
constructs this wavefunction with something known as a "Slater
determinant":

.. math::

    \Psi(\textbf{r}_1, \textbf{r}_2, \ ... , \textbf{r}_N) = \frac{1}{\sqrt{ N!}}
   \begin{vmatrix}
   \psi_1(\textbf{r}_1) & \psi_1(\textbf{r}_2)  & \dots & \psi_1(\textbf{r}_N) \\
   \psi_2(\textbf{r}_1) & \psi_2(\textbf{r}_2)  & \dots & \psi_2(\textbf{r}_N) \\
   \vdots & \vdots & \ddots & \vdots\\
   \psi_N(\textbf{r}_1) & \psi_N(\textbf{r}_2)  & \dots & \psi_N(\textbf{r}_N)
   \end{vmatrix} .

Unfortunately, but unsurpringsingly, making this approximation means
that you lose a lot of the physics in the problem (look at the double
antisymmetric well notebook for a prime example). Generally, in this
system electrons spend more time closer to each other than they would if
the repulsion between them had been taken into account. This is often a
useful thing to calculate, however, since the code runs quickly and can
then provide a reasonable first guess for the density for a
self-consistant field approach - more on this later.
