Adding to iDEA
==============

The version control system used to manage the iDEA source code is `git <https://git-scm.com/>`_. Git offers a "learn git in 15 minutes" tutorial found `here <https://try.github.io/>`_. 

Commiting changes locally
-------------------------

Once you have made a change to a file, you will want to commit this change to your local repository. This
will ensure that as you pull changes from the central repository they will be automatically integrated into your work.
To see the list of files you have changed run

.. code-block:: bash

   git status

Then to add a file you want to commit run

.. code-block:: bash

   git add file_name

Once you have finished adding files you can commit your changes locally using

.. code-block:: bash

   git commit

You will be prompted to enter a commit message to describe your changes and save the file. Your changes are now committed!


Pulling your changes into the central repository
------------------------------------------------

To have the changes you have commited pulled into the central repository for everyone to access, email jw1294@york.ac.uk with a pull request.
It is a good idea to run the unit tests before makeing a pull request to make sure nothing had broken.


Pulling changes from the central repository
-------------------------------------------

Once another user has had their changes pulled into the central repository you will want to fetch 
these changes into your local repository and merge them with your work. To do this run

.. code-block:: bash

   git pull

You will not be able to perfrom this pull if you have untracked changes, you should first commit your changes as described above.
If you do not wish to commit the untracked changes and simply want to remove them run

.. code-block:: bash

   git stash



