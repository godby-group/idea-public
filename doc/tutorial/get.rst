Getting iDEA
============


Installation requirements
-------------------------

 * `Python <http://www.python.org>`_ 2.7 or later
 * `numpy <http://www.numpy.org>`_ 1.10 or later
 * `scipy <http://www.scipy.org>`_ 0.17 or later
 * *(optional)* `Intel MKL  <https://software.intel.com/en-us/intel-mkl>`_ 11.3
   or later for parallel execution

 * *(optional)* `Sphinx <http://sphinx-doc.org>`_ 1.4 or later for generating
   the documentation

 * *(optional)* `matplotlib <http://matplotlib.org/>` 1.4.3 or later for post-processing

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

