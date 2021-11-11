Installation on Linux (>=20.04.03)
**********************************

.. toctree::
   :maxdepth: 3

This section of our documentation is designed to help you get started
deploying a PyGrid Domain to Linux with a version higher or equal to ``20.04.03`` in the simplest way possible.

.. seealso::

   For more advanced tutorials, such as cloud deployment, ansible, vagrant, kubernetes, or virtualbox deployment, please check
   `advanced deployment documentation <https://openmined.github.io/PySyft/deployment/index.html#>`__.
.. seealso::

We will use the Linux Terminal to install all the prerequisites and launch the domain. A quick way to launch the terminal is by pressing ``Ctrl+Alt+T``. Let's go!

1. **Installing Python 3.9**

For the rest of the tutorial, you need to have installed at least Python 3.9.
If you think you might have it already installed, you can run the following to check:

.. code-block:: bash

   $ python3 --version
   Python 3.9.0

If it doesn't show the correct version, we can install it by running the following:

.. code-block:: bash

   $ sudo apt update
   $ sudo apt install python3.9
   $ python3 --version
   Python 3.9.0

2. **Installing and using Pip**

`Pip <https://pip.pypa.io/en/stable/>`__ is the most used package installer for Python and will help us to install dependencies required much easier.
You can install it by running the following:

.. code-block:: bash

   $ python -m ensurepip --upgrade

If you have it already installed, ensure it is the latest version by running:

.. code-block:: bash

   $ python -m pip install --upgrade pip


3. **Conda and setting up a virtual environment**

Conda is a package manager that helps you to easily install a lot of data science and machine learning packages, but also to create a separated environment when a certain set of dependencies need to be installed.
To install Conda, you can:

    a. Download the `Anaconda installer <https://www.anaconda.com/products/individual#linux>`__.

    b. Run the following depending where you downloaded the installer (e.g. `~/Downloads/`):

        .. code-block:: bash

           $ bash ~/Downloads/Anaconda3-2020.02-Linux-x86_64.sh

        .. note::

           Please note that the naming might be different given it could be a newer version of Anaconda. This is ok, you can proceed further.

    c. Create a new env specifying the Python version (we recommend Python 3.8/3.9) in the terminal:

        .. code-block:: bash

           $ conda create -n myenv python=3.9
           $ conda activate myenv
           (to exit): conda deactivate

4. **Install Jupyter Notebook**

A very convenient way to interact with a deployed node is via Python, using a Jupyter Notebook. You can install it by running:

.. code-block:: bash

  $ pip install jupyter-notebook

If you encounter issues, you can also install it using Conda:

.. code-block:: bash

  $ conda install -c conda-forge notebook

To launch the Jupyter Notebook, you can run the following in your terminal:

.. code-block:: bash

  $ jupyter notebook

5. **Installing and configuring Docker**

`Docker <https://docs.docker.com/get-started/overview/>`__ is a framework which allows us to separate the infrastructure needed to run PySyft in an isolated environment called ``container`` which you can use off the shelf, without many concerns.
If it sounds complicated, please mind with us and we will quickly do the required steps!
Additionally, we will also use `Docker Composite V2 <https://docs.docker.com/compose/>`_, which allows us to run multi-container applications.


a. Install **Docker**:

    .. code-block:: bash

       $ sudo apt-get upgrade docker & docker run hello-world

b. Install **Docker Composite V2** as described `here <https://docs.docker.com/compose/cli-command/#installing-compose-v2>`__.

c. You should see ‘Docker Compose version v2’ when running:

    .. code-block:: bash

       $ docker compose version
       Docker Compose version v2

d. If not, go through the `instructions here <https://www.rockyourcode.com/how-to-install-docker-compose-v2-on-linux-2021/>`__ or if you are using Linux, you can try to do:

    .. code-block:: bash

       $ mkdir -p ~/.docker/cli-plugins
       $ curl -sSL https://github.com/docker/compose-cli/releases/download/v2.0.0-beta.5/docker-compose-linux-amd64 -o ~/.docker/cli-plugins/docker-compose
       $ chmod +x ~/.docker/cli-plugins/docker-compose

e. Also, make sure you can run without sudo:

    .. code-block:: bash

       $ echo $USER //(should return your username)
       $ sudo usermod -aG docker $USER

6. **Install PySyft and PyGrid**

The hardest part is done! To install the OpenMined stack that you need in order to deploy a node, please run:

.. code-block:: bash

   $ pip install --pre syft
   $ pip install hagrid=0.1.8

Syft is the library which contains the tools to run privacy preserving machine learning.
Hagrid is a commandline tool that speeds up the deployment of PyGrid, the provider of a peer-to-peer network of
data owners and data scientists who can collectively train AI model using Syft.

7. **Launch the Doman Node**

Congrats for making it that far! One last step to unleash the power of Hagrid!
To launch the domain node, you can run:

.. code-block:: bash

   $ hagrid launch domain to docker:8081

To stop the running node, you can run:

.. code-block:: bash

   $ hagrid land

But before stopping it, you can go to ```localhost:8081`` in your `broswer <localhost:8081>`_ to actually interact with the PyGrid Admin UI, where you can manage as a Data Owner your datasets, as well as incoming requests from data scientist.
You can log in using the following credentials:

.. code-block:: python

   info@openmined.org

   changethis

Now you're all set up to fully start using PySyft!
