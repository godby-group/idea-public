iDEA ELF (Electron Localisation Function)
=========================================

The electron localisation function (ELF) is an approximate method to estimate the probability of finding an electron in the vicinity of a reference electron i.e. electron localisation is the tendency of an electron in a many-electron system to exclude other electrons from its vicinity. This is useful, for example, in describing chemical bonds, which are formed by pairs of localised electrons.

The original definition of ELF (in 1D) is given by:

.. math:: \mathrm{ELF}(x) = \frac{1}{1 + \bigg(\frac{D_{\sigma}(x)}{D_{\sigma, \mathrm{H}}(x)}\bigg)^{2}},

where :math:`D_{\sigma}(x)` characterizes the probability of finding a second electron close to the reference electron, and :math:`D_{\sigma, \mathrm{H}}(x)` is the same quantity but for the homogeneous electron gas (HEG). In our code, we provide different ways of calculating ELF(x).


Dobson (exact) ELF
------------------

The ELF developed by Dobson calculates ELF(x) exactly and requires the many-electron wavefunction, which is possible for 2 or 3 electron systems:

.. math:: D_{\sigma}(x) = \frac{\big[\nabla_{x'}^{2}n_{2}(x,x')\big]_{x'=x}}{2n(x)},

where :math:`n_{2}(x,x')` is the electron pair density.

Becke-Edgecombe (approximate) ELF
---------------------------------

In practical calculations the many-electron wavefunction cannot be calculated and so :math:`D_{\sigma}(x)`, and hence :math:`\mathrm{ELF}(x)`, have to be approximated. Becke and Edgecombe approximated :math:`D_{\sigma}(x)` as:

.. math:: D_{\sigma}(x) \approx \sum\limits_{i}^{N_{\sigma}}|\nabla \phi_{i}^{\sigma}(x)|^{2} - \frac{1}{4}\frac{\left|\nabla n_{\sigma}(x)\right|^{2}}{n_{\sigma}(x)},

where :math:`\{\phi_{i}\}` are single-particle states with an electron density :math:`n_{\sigma}`.
