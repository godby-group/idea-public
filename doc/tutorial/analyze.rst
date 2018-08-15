Running ViDEO
=============

ViDEO is a python script used to process the raw output files generared by iDEA into
easy to use .dat data files, .pdf plots and .mp4 animations

Using ViDEO directly
--------------------

Simply naviage to the relevent outputs folder

.. code-block:: bash

   cd outputs/run_name/

and run the script

.. code-block:: bash

   python ViDEO.py

You will be promted for information to determine the result you want to process, and what files
should be generated. ViDEO makes use of the file convention of outpus used in iDEA, as all pickle
files are names according to the following convention

.. code-block:: bash

   {gs/td}_{appoximation}_{quantity}.db

for example, the ground-state non-interacting density

.. code-block:: bash

   gs_non_den.db


Using ViDEO on results generated from a script
----------------------------------------------

Simply naviage to the relevent outputs folder (assuming :code:`run.save = True` in the relevent input file)

.. code-block:: bash

   cd outputs/run_name/

and run the command

.. code-block:: bash

   ViDEO.py
