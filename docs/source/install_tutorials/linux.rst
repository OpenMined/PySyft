.. _linux_install:

==================================
Installation on Linux (>=20.04.03)
==================================

.. toctree::
   :maxdepth: 3

This documentation is to help you install and be able to deploy a Domain Node on Ubuntu Linux, with a version of ``20.04.03`` or newer, in the simplest way possible.

.. seealso::

   For more advanced tutorials, such as cloud deployment, ansible, vagrant, kubernetes, or virtualbox deployment, please check
   `advanced deployment documentation <https://openmined.github.io/PySyft/deployment/index.html#>`__.

We will use the Linux Terminal to install all the prerequisites and launch the domain. A quick way to launch the terminal is by pressing ``Ctrl+Alt+T``. Let's go!

1. **Installing Python 3.9**

We'll be working with Python 3.9 or newer. To check if you have it installed, you may run:

.. code-block:: bash

   python3 --version

If you don't have the correct version of Python, installing it is as easy as running the following:

.. code-block:: bash

   sudo apt update
   sudo apt install python3.9
   python3 --version

2. **Installing and using Pip**

`Pip <https://pip.pypa.io/en/stable/>`__ is the most widely used package installer for Python and will help us to install the required dependencies MUCH easier.
You can install it by running the following:

.. code-block:: bash

   python -m ensurepip --upgrade

If you already have it installed, you can check to make sure it's the latest version by running:

.. code-block:: bash

   python -m pip install --upgrade pip


3. **Conda and setting up a virtual environment**

Conda is a package manager that helps you to easily install a lot of data science and machine learning packages, but also to create a separated environment when a certain set of dependencies need to be installed.
To install Conda, you can:

    a. Download the `Anaconda installer <https://www.anaconda.com/products/individual#Downloads>`__.

    b. Run the following code, modifying it depending on where you downloaded the installer (e.g. `~/Downloads/`):

        .. code-block:: bash

           bash ~/Downloads/Anaconda3-2020.02-Linux-x86_64.sh

        .. note::

           Please note that the naming might be different given it could be a newer version of Anaconda. This is ok, you can proceed further.

    c. Create a new env specifying the Python version (we recommend Python 3.8/3.9) in the terminal:

        .. code-block:: bash

           conda create -n syft_env python=3.9
           conda activate syft_env


    d. To exit, you might run:

        .. code-block:: bash

           conda deactivate

4. **Install Jupyter Notebook**

A very convenient way to interact with a deployed node is via Python, using a Jupyter Notebook. You can install it by running:

.. code-block:: bash

  pip install jupyter-notebook

If you encounter issues, you can also install it using Conda:

.. code-block:: bash

  conda install -c conda-forge notebook

To launch the Jupyter Notebook, you can run the following in your terminal:

.. code-block:: bash

  jupyter notebook

5. **Installing and configuring Docker**

`Docker <https://docs.docker.com/get-started/overview/>`__ is a framework which allows us to separate the infrastructure needed to run PySyft in an isolated environment called a ``container`` which you can use off the shelf, without many concerns.
If it sounds complicated, please don't worry- we will walk you through all steps, and you'll be done in no time!
Additionally, we will also use `Docker Composite V2 <https://docs.docker.com/compose/>`_, which allows us to run multi-container applications.


a. Install **Docker**:

    .. code-block:: bash

       sudo apt-get upgrade docker & docker run hello-world

b. Install **Docker Composite V2** as described `here <https://docs.docker.com/compose/cli-command/#installing-compose-v2>`__.

c. You should see ‘Docker Compose version v2’ when running:

    .. code-block:: bash

       docker compose version
       Docker Compose version v2

d. If not, go through the `instructions here <https://www.rockyourcode.com/how-to-install-docker-compose-v2-on-linux-2021/>`__ or if you are using Linux, you can try to do:

    .. code-block:: bash

       mkdir -p ~/.docker/cli-plugins
       curl -sSL https://github.com/docker/compose/releases/download/v2.2.3/docker-compose-linux-x86_64 -o ~/.docker/cli-plugins/docker-compose
       chmod +x ~/.docker/cli-plugins/docker-compose

e. Also, make sure you can run without sudo:

    .. code-block:: bash

       echo $USER //(should return your username)
       sudo usermod -aG docker $USER

6. **Install PySyft and PyGrid**

The hardest part is done! To install the OpenMined stack that you need in order to deploy a node, please run:

.. code-block:: bash

   $ pip install syft
   $ pip install hagrid


PySyft is a library which contains the tools to run privacy preserving machine learning.
Hagrid is a commandline tool that speeds up the deployment of PyGrid, the provider of a peer-to-peer network of
data owners and data scientists who can collectively train AI model using Syft.

7. **Launch the Domain Node**

Congrats for making it this far! You only have one final step remaining, before you unleash the power of Hagrid!
The final step is to launch a domain node, which is as easy as:

.. code-block:: bash

   hagrid launch

To stop the running domain,
Find the auto-assigned name of your domain (look at the lines just after you executed the above command) and run:

.. code-block:: bash

   hagrid land <name_of_domain>

But before stopping it, you can go to ```localhost:8081`` in your `browser <localhost:8081>`_ to actually interact with the PyGrid Admin UI, where you can manage as a Data Owner your datasets, as well as incoming requests from data scientist.
You can log in using the following credentials:

.. code-block:: python

   info@openmined.org

   changethis

Now you're all set up to fully start using PySyft!
