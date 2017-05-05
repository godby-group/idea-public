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

 * *(optional)* `Sphinx <http://sphinx-doc.org>`_ 1.4 or later for generating
   the documentation

 * *(optional)* `matplotlib <http://matplotlib.org/>`_ 1.4.3 or later for post-processing

Installing iDEA
----------------

.. code-block:: bash

   git clone user@rwgu1.york.ac.uk:~jw1294/iDEAL my_idea
   cd my_idea

   # add current working directory to PYTHONPATH
   echo "export PYTHONPATH=\$PYTHONPATH:`pwd`" >> ~/.bashrc
   # add scripts directory to your PATH
   echo "export PATH=\$PATH:`pwd`/../scripts" >> ~/.bashrc
   source ~/.bashrc

iDEA includes some Fortran extensions, which are built automatically the first
time you run it. If the default options for the compiler, libraries etc.  do
not work for your platform, you will need to adapt the options for one of the
available architectures in :code:`iDEA/arch/<your architecture>.mk`. Then 
:code:`export ARCH=<your architecture>` in order to tell iDEA to use this
architecture file.

Generating documentation
------------------------

.. code-block:: bash

   # build documentation in doc/_build/html
   cd doc
   bash make_doc.sh  



Updating iDEA
-------------
To update your working copy to the latest version, simply do

.. code-block:: bash

   # Pull all changes from central git repository
   git pull

