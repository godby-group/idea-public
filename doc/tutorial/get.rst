Getting iDEA
============


Installation requirements
-------------------------

 * `Python <http://www.python.org>`_ 2.7 or later
 * `numpy <http://www.numpy.org>`_ 1.10 or later
 * `scipy <http://www.scipy.org>`_ 0.17 or later
 * *(optional)* Parallel execution requires the 
   `Intel MKL  <https://software.intel.com/en-us/intel-mkl>`_ 11.3 or later
 * *(optional)* Generating the documentation requires
   `Sphinx <http://sphinx-doc.org>`_ 1.4 or later

Installing iDEA
----------------

.. code-block:: bash

   git clone user@rwgu1.york.ac.uk:~jw1294/iDEAL .
   git checkout v2.0  # move to current development branch
   # add current working directory to PYTHONPATH
   echo "export PYTHONPATH=\$PYTHONPATH:`pwd`" >> ~/.bashrc
   # add scripts directory to your PATH
   echo "export PATH=\$PATH:`pwd`/scripts" >> ~/.bashrc
   source ~/.bashrc

   cd iDEA
   make  # makes Fortran MKL libraries

Generating documentation
------------------------

.. code-block:: bash

   # build documentation in doc/_build/html
   cd doc
   bash make_doc.sh  

Run unit tests
--------------

.. code-block:: bash

   # run this in the base directory
   python -m unittest discover
