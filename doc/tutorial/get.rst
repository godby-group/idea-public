Getting iDEA
============


Installation requirements
-------------------------

 * `Python <http://www.python.org>`_ 2.7 or later
 * `numpy <http://www.numpy.org>`_ ... or later
 * `scipy <http://www.scipy.org>`_ ... or later
 * *(optional)* Parallel execution requires the 
   `Intel MKL  <https://software.intel.com/en-us/intel-mkl>`_ 11.3 or later
 * *(optional)* Generating the documentation requires
   `Sphinx <http://sphinx-doc.org>`_ 1.3.3 or later

Installing iDEA
----------------

.. code-block:: bash

   scp -r user@rwgu1.york.ac.uk:~jw1294/iDEAL .
   cd iDEAL
   # add current working directory to PYTHONPATH
   mycwd=`pwd`
   echo "export PYTHONPATH=\$PYTHONPATH:$mycwd" >> ~/.bashrc

Generating documentation
------------------------

.. code-block:: bash

   cd iDEAL/doc
   bash apidoc.sh    # parses iDEA python modules
   make html         # builds HTML documentation
