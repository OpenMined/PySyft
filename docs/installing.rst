**************
Install PySyft
**************

This page shows how to install Syft version 0.3.0. As this software is currently in
alpha, there are changes happening every day. Thus, the best way to install PySyft is
to build it from source. Each of the tutorials below describe how to build PySyft from
source within each respective operating system.

If you get stuck anywhere in this tutorial, please `join OpenMined's slack <https://slack.openmined.org>`_
and ask a question in the channel called `#topic_beginner_questions <https://openmined.slack.com/archives/C6DEWA4FR>`_.

.. note::

    Want to make a youtube video demonstrating this installation on your machine? We'd
    be very grateful if you made one! Just create a Pull Request adding a link to your
    video to this documentation. If you don't know how to do that, just share your
    video in `#topic_beginner_questions <https://openmined.slack.com/archives/C6DEWA4FR>`_
    in `OpenMined's slack <https://slack.openmined.org>`_.

Mac OS X Installation
=====================

The following are the instructions for how to build PySyft from source if you use the
OSX operating system.


Step 1 - Clone PySyft Repository
--------------------------------

First, assuming you're in a directory where you're comfortable downloading PySyft,
run the following command.

.. code:: console

    git clone https://github.com/OpenMined/PySyft.git
    cd PySyft
    ls

You should now see the current folder directory of PySyft, matching what you see
at https://github.com/OpenMined/PySyft. Something like the following:

.. code:: console

    > ls
    AUTHORS.rst      LICENSE.txt      __pycache__      examples         scripts          src
    CHANGELOG.rst    README.md        build            proto            setup.cfg        tests
    CONTRIBUTING.md  __init__.py      docs             requirements.txt setup.py         untitled.md


Step 2 - Check Python Version
-----------------------------

PySyft can only run on python versions 3.6 and up. At the time of writing, this is only
python versions 3.6, 3.7, and 3.8. So, before you proceed, you need to ensure that you
are running one of these versions. Run the following:

.. code:: console

    > python --version
    Python 3.8.1

If the version printed underneath your command is less than 3.6, then you MUST use
anaconda in the next step (which is recommended anyway). Technically, you could try to
upgrade your version of python but fair warning... e're be dragons.


Step 3 - Setup Environment
--------------------------

I'm sure you're tempted to skip this step. Word from the wise...
`Don't skip this step.
<https://twitter.com/iamtrask/status/1300854373296332809>`_

You are about to install a library with lots of complex dependencies. You don't want to
break something on your computer because you're installing PySyft. And vice versa, you
don't want to later break your PySyft install when installing some other tool later!
Friends don't let friends build libraries from source without using a virtual
environment.

And since PySyft uses (and will use more and more) non-python dependencies, the best
virtual environment to use for PySyft is conda. Note, if you're tempted to use
virtualenv instead, `read this warning <https://twitter.com/shreyshahi/status/1300855906742140928>`_.

Step 3.1 - Install Conda
^^^^^^^^^^^^^^^^^^^^^^^^

First, let's see if you have conda installed! Type "conda" into your Terminal app and
hit enter.

.. code:: console

    > conda
    usage: conda [-h] [-V] command ...

    conda is a tool for managing and deploying applications, environments and packages.

    Options:

    positional arguments:
      command
        clean        Remove unused packages and caches.
        config       Modify configuration values in .condarc. This is modeled
                     after the git config command. Writes to the user .condarc
                     file (/Users/atrask/.condarc) by default.
        create       Create a new conda environment from a list of specified

If calling "conda" doesn't return something like this, then you need to install conda.
Just follow the `installation instructions <https://docs.conda.io/projects/conda/en/latest/user-guide/install/>`_
and you'll be fine.

Step 3.2 - Create conda Env
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Now, we want to create a conda virtual environment with the latest verison of Python
supported by syft which, at the time of writing, is 3.8.

.. code:: console

    conda create -n my_syft_env python=3.8

Then follow the instructions it gives you to create your environment.


Step 3.3 - Activate Conda Env
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To activate the environment you created in the last step, simply enter
`activate <environment name>` which if you simply copy pasted the line above, the
environment name was "my_syft_env".

.. code:: console

    conda activate my_syft_env

.. note::

    You will need to activate this my_syft_env environment whenever you want to use
    syft, unless of course you install syft in another environment.


Step 4 - Install Dependencies
-----------------------------

Assuming you're still in the base directory of PySyft (as you came to be in step 1),
you can now install the dependencies that PySyft relies on using the following command:

.. code:: console

    pip install -r requirements.txt

This should install all the libraries you need for PySyft. Just in case, let's make
sure you have a compatible version of PyTorch. Your PyTorch version should be 1.3 or
later. Open up a python shell (by running `python` in your Terminal client) and then
execute the following python code.

.. code:: python

    import torch
    print(torch.__version__)

As long as this reads 1.3 or later, you should be good. If it's 1.3 or earlier, then
upgrade it by installing the latest version.

.. code:: console

    pip install torch -U

Step 5 - Install PySyft
-----------------------

You are now ready to install PySyft! We recommend the following just in case you later
decide you want to help edit the codebase.

.. code:: python

    pip install -e .

This will create a permanent pointer from the PySyft code to your conda VM. That means
that if you make any changes to the code you won't have to re-install PySyft to be able
to use them! If you don't want this fanciness you can also run the good ole fashioned
setup.py install.

.. code:: python

    python setup.py install

Step 6 - Run Light Tests
------------------------

At the time of writing, we have quite a few unit tests but most of them are simply
testing the PyTorch runtime. To run the abbreviated set of tests (and make sure your
installation is happy), run the following.

.. code:: python

    pytest -k "not test_all_allowlisted_tensor_methods_work_remotely_on_all_types"

(If you don't have pytest installed, run "pip install pytest")

Optional - Run All Tests
------------------------

If you'd like to run the full test suite, you may do so by running the following

.. code:: python

    python setup.py test

Congratulations! You have just installed PySyft on Mac OSX!

Linux Installation
==================

The following are the instructions for how to build PySyft from source if you use the
Linux operating system.

Step 1 - Clone PySyft Repository
--------------------------------

First, assuming you're in a directory where you're comfortable downloading PySyft,
run the following command.

.. code:: console

    git clone https://github.com/OpenMined/PySyft.git
    cd PySyft
    ls

You should now see the current folder directory of PySyft, matching what you see
at https://github.com/OpenMined/PySyft. Something like the following:

.. code:: console

    > ls
    AUTHORS.rst      LICENSE.txt      __pycache__      examples         scripts          src
    CHANGELOG.rst    README.md        build            proto            setup.cfg        tests
    CONTRIBUTING.md  __init__.py      docs             requirements.txt setup.py         untitled.md

Step 2 - Check Python Version
-----------------------------

PySyft can only run on python versions 3.6 and up. At the time of writing, this is only
python versions 3.6, 3.7, and 3.8. So, before you proceed, you need to ensure that you
are running one of these versions. Run the following:

.. code:: console

    > python --version
    Python 3.8.1

If the version printed underneath your command is less than 3.6, then you MUST use anaconda
in the next step (which is recommended anyway). Technically, you could try to upgrade your
version of python but fair warning... e're be dragons.

Step 3 - Setup Environment
--------------------------

I'm sure you're tempted to skip this step. Word from the wise...
`Don't skip this step.
<https://twitter.com/iamtrask/status/1300854373296332809>`_

You are about to install a library with lots of complex dependencies. You don't want to break
something on your computer because you're installing PySyft. And vice versa, you don't want
to later break your PySyft install when installing some other tool later! Friends don't
let friends build libraries from source without using a virtual environment.

And since PySyft uses (and will use more and more) non-python dependencies, the best
virtual environment to use for PySyft is conda. Note, if you're tempted to use virtualenv
instead, `read this warning <https://twitter.com/shreyshahi/status/1300855906742140928>`_.

Step 3.1 - Install Conda
^^^^^^^^^^^^^^^^^^^^^^^^

First, let's see if you have conda installed! Type "conda" into your Terminal app and hit enter.

.. code:: console

    > conda
    usage: conda [-h] [-V] command ...

    conda is a tool for managing and deploying applications, environments and packages.

    Options:

    positional arguments:
      command
        clean        Remove unused packages and caches.
        config       Modify configuration values in .condarc. This is modeled
                     after the git config command. Writes to the user .condarc
                     file (/Users/atrask/.condarc) by default.
        create       Create a new conda environment from a list of specified

If calling "conda" doesn't return something like this, then you need to install conda. Just
follow the `installation instructions <https://docs.conda.io/projects/conda/en/latest/user-guide/install/>`_
and you'll be fine.

Step 3.2 - Create conda Env
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Now, we want to create a conda virtual environment with the latest verison of Python supported
by syft which, at the time of writing, is 3.8.

.. code:: console

    conda create -n my_syft_env python=3.8

Then follow the instructions it gives you to create your environment.

Step 3.3 - Activate Conda Env
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To activate the environment you created in the last step, simply enter `activate <environment name>`
which if you simply copy pasted the line above, the environment name was "my_syft_env".

.. code:: console

    conda activate my_syft_env

.. note::

    You will need to activate this my_syft_env environment whenever you want to use syft,
    unless of course you install syft in another environment.

Step 4 - Install Dependencies
-----------------------------

Assuming you're still in the base directory of PySyft (as you came to be in step 1), you can
now install the dependencies that PySyft relies on using the following command:

.. code:: console

    pip install -r requirements.txt

This should install all the libraries you need for PySyft. Just in case, let's make sure
you have a compatible version of PyTorch. Your PyTorch version should be 1.3 or later. Open
up a python shell (by running `python` in your Terminal client) and then execute the following
python code.

.. code:: python

    import torch
    print(torch.__version__)

As long as this reads 1.3 or later, you should be good. If it's 1.3 or earlier, then upgrade
it by installing the latest version.

.. code:: console

    pip install torch -U

Step 5 - Install PySyft
-----------------------

You are now ready to install PySyft! We recommend the following just in case you later decide
you want to help edit the codebase.

.. code:: python

    pip install -e .

This will create a permanent pointer from the PySyft code to your conda VM. That means that if
you make any changes to the code you won't have to re-install PySyft to be able to use them!
If you don't want this fanciness you can also run the good ole fashioned setup.py install.

.. code:: python

    python setup.py install

Step 6 - Run Light Tests
------------------------

At the time of writing, we have quite a few unit tests but most of them are simply testing
the PyTorch runtime. To run the abbreviated set of tests (and make sure your installation
is happy), run the following.

.. code:: python

    pytest -k "not test_all_allowlisted_tensor_methods_work_remotely_on_all_types"

(If you don't have pytest installed, run "pip install pytest")

Optional - Run All Tests
------------------------

If you'd like to run the full test suite, you may do so by running the following

.. code:: python

    python setup.py test

Congratulations! You have just installed PySyft on Linux !


Windows Installation
====================

The following are the instructions for how to build PySyft from source if you use the
Windows operating system.

Step 1 - Install Git
--------------------

Here is the download link for Git on windows:  `Git for Windows <https://gitforwindows.org/>`_
or in case you are lazy! you can use  `Github for Desktop. <https://desktop.github.com/>`_

Step 2 - Install Microsoft Build tools
--------------------------------------

Go to the `Download page <https://visualstudio.microsoft.com/downloads/>`_ and click on `Free download` under **Community** in the Visual Studio download section. 

After the download is finished, run the downloaded package. In the installation window select `Desktop development with C++` and click on `Install` at the bottom-right corner of the page. (In the above screenshot you see a `Close` button instead since I have already installed it.)

Wait until the the installation has finished. (Have a break as it will take time! 😪)


Step 3 - Clone PySyft Repository
--------------------------------

First, assuming you're in a directory where you're comfortable downloading PySyft,
run the following command.

.. code:: console

    git clone https://github.com/OpenMined/PySyft.git
    cd PySyft
    ls

You should now see the current folder directory of PySyft, matching what you see
at https://github.com/OpenMined/PySyft. Something like the following:

.. code:: console

    > ls
    AUTHORS.rst      LICENSE.txt      __pycache__      examples         scripts          src
    CHANGELOG.rst    README.md        build            proto            setup.cfg        tests
    CONTRIBUTING.md  __init__.py      docs             requirements.txt setup.py         untitled.md

Step 4 - Check Python Version
-----------------------------

PySyft can only run on python versions 3.6 and up. At the time of writing, this is only
python versions 3.6, 3.7, and 3.8. So, before you proceed, you need to ensure that you
are running one of these versions. Run the following:

.. code:: console

    > python --version
    Python 3.8.1

If the version printed underneath your command is less than 3.6, then you MUST use anaconda
in the next step (which is recommended anyway). Technically, you could try to upgrade your
version of python but fair warning... e're be dragons.

Step 5 - Setup Environment
--------------------------

I'm sure you're tempted to skip this step. Word from the wise...
`Don't skip this step.
<https://twitter.com/iamtrask/status/1300854373296332809>`_

You are about to install a library with lots of complex dependencies. You don't want to break
something on your computer because you're installing PySyft. And vice versa, you don't want
to later break your PySyft install when installing some other tool later! Friends don't
let friends build libraries from source without using a virtual environment.

And since PySyft uses (and will use more and more) non-python dependencies, the best
virtual environment to use for PySyft is conda. Note, if you're tempted to use virtualenv
instead, `read this warning <https://twitter.com/shreyshahi/status/1300855906742140928>`_.

Step 5.1 - Install Conda
^^^^^^^^^^^^^^^^^^^^^^^^

First, let's see if you have conda installed! Type "conda" into your Terminal app and hit enter.

.. code:: console

    > conda
    usage: conda [-h] [-V] command ...

    conda is a tool for managing and deploying applications, environments and packages.

    Options:

    positional arguments:
      command
        clean        Remove unused packages and caches.
        config       Modify configuration values in .condarc. This is modeled
                     after the git config command. Writes to the user .condarc
                     file (/Users/atrask/.condarc) by default.
        create       Create a new conda environment from a list of specified

If calling "conda" doesn't return something like this, then you need to install conda. Just
follow the `installation instructions <https://docs.conda.io/projects/conda/en/latest/user-guide/install/>`_
and you'll be fine.

Step 5.2 - Create conda Env
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Now, we want to create a conda virtual environment with the latest verison of Python supported
by syft which, at the time of writing, is 3.8.

.. code:: console

    conda create -n my_syft_env python=3.8

Then follow the instructions it gives you to create your environment.

Step 5.3 - Activate Conda Env
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To activate the environment you created in the last step, simply enter `activate <environment name>`
which if you simply copy pasted the line above, the environment name was "my_syft_env".

.. code:: console

    conda activate my_syft_env

.. note::

    You will need to activate this my_syft_env environment whenever you want to use syft,
    unless of course you install syft in another environment.

Step 6 - Install Dependencies
-----------------------------

Assuming you're still in the base directory of PySyft (as you came to be in step 1), you can
now install the dependencies that PySyft relies on using the following command:

.. code:: console

    pip install -r requirements.txt

This should install all the libraries you need for PySyft. Just in case, let's make sure
you have a compatible version of PyTorch. Your PyTorch version should be 1.3 or later. Open
up a python shell (by running `python` in your Terminal client) and then execute the following
python code.

.. code:: python

    import torch
    print(torch.__version__)

As long as this reads 1.3 or later, you should be good. If it's 1.3 or earlier, then upgrade
it by installing the latest version.

.. code:: console

    pip install torch -U
    
or else you can get the installation command from `here. <https://pytorch.org/get-started/locally/>`_ (use the pip option)

Step 7 - Install PySyft
-----------------------

You are now ready to install PySyft! We recommend the following just in case you later decide
you want to help edit the codebase.

.. code:: python

    pip install -e .

This will create a permanent pointer from the PySyft code to your conda VM. That means that if
you make any changes to the code you won't have to re-install PySyft to be able to use them!
If you don't want this fanciness you can also run the good ole fashioned setup.py install.

.. code:: python

    python setup.py install
    
Step 6 - Run Light Tests
------------------------

At the time of writing, we have quite a few unit tests but most of them are simply
testing the PyTorch runtime. To run the abbreviated set of tests (and make sure your
installation is happy), run the following.

.. code:: python

    pytest -k "not test_all_allowlisted_tensor_methods_work_remotely_on_all_types"

(If you don't have pytest installed, run "pip install pytest")

Optional - Run All Tests
------------------------

If you'd like to run the full test suite, you may do so by running the following

.. code:: python

    python setup.py test

Congratulations! You have just installed PySyft on Windows!

