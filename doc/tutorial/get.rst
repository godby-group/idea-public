Getting iDEA
============


Installation requirements
-------------------------

 * `Python <http://www.python.org>`_ 2.7 or later
 * `numpy <http://www.numpy.org>`_ 1.10 or later
 * `scipy <http://www.scipy.org>`_ 0.17 or later
 * Fortran90 compiler, such as `gfortran <https://gcc.gnu.org/fortran/>`_ 5 or
   later, `ifort <https://software.intel.com/en-us/fortran-compilers>`_ 14 or
   later
 * *(optional)* `Intel MKL  <https://software.intel.com/en-us/intel-mkl>`_ 11.3
   or later for parallel execution
 * *(optional)* `matplotlib <http://matplotlib.org/>`_ 1.4.3 or later for post-processing

Installing iDEA
----------------

.. code-block:: bash

   git clone USERNAME@rwgu4.its.york.ac.uk:/shared/storage/physrwg/trunk/iDEAL/ my_idea
   cd my_idea

   # install iDEA for your unix user, including documentation extension
   pip install --user -e .[doc]

iDEA includes some Fortran extensions, which are built automatically the first
time you run it. If the default options for the compiler, libraries etc.  do
not work for your platform, you will need to adapt the options for one of the
available architectures in :code:`iDEA/arch/<your architecture>.mk`. Then
:code:`export ARCH=<your architecture>` in order to tell iDEA to use this
architecture file.

Updating iDEA
-------------

.. code-block:: bash

   # Pull all changes from central git repository
   git pull

.. _generate-documentation:

Generating the documentation
-----------------------------
A recent version of the documentation can be found on the iDEA web page.
If you are making changes to the code and/or the documentation, you may
need to generate the documentation by yourself

**Requirements**

 * `Sphinx <http://sphinx-doc.org>`_ 1.4 or later 
 * `numpydoc extension <https://pypi.python.org/pypi/numpydoc>`_ 0.7 or later (adds support for numpy-style docstrings)
 * `nbconvert extension <http://sphinx-doc.org>`_ 5.2 or later (renders static versions of jupyter notebooks)
 * (optional) `LaTeX <https://www.latex-project.org/get/>`_ (for the LaTeX version of the documentation)

Note: in order to install the required packages on a system without admin rights, do

.. code-block:: bash

   pip install --user sphinx numpydoc nbconvert

In order to produce the documentation in html and latex form:

.. code-block:: bash

   cd doc
   bash make_doc.sh
   # find html documentation in _build/html
   # find latex documentation in _build/latex 
   make latexpdf  # generates _build/latex/iDEA.pdf


