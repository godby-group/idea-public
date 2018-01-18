.. _practises:

Coding practises
================

If you'd like to contribute your changes back to iDEA,
please follow good coding practises.

Write unit tests
----------------

For any new feature you add, make sure to add a 
**unit test** that checks you feature is working as intended.

 * Naming convention: :code:`test_<your_module>.py`
 * start by copying a simple example, e.g. :code:`test_NON.py`
 * make sure your test is quick,
   it should run *in the blink of an eye*
   
Checklist
---------

Before making a pull request, go through the following checklist:

 1. Check that you didn't break the existing unit tests:

    .. code-block:: bash

       # run this in the base directory
       python -m unittest discover

 2. Check that the documentation builds fine:

    .. code-block:: bash

       cd doc/
       bash make_doc.sh

 3. Check the test coverage for your module
    (should be close to 100%):

    .. code-block:: bash

       firefox doc/_build/coverage/index.html
