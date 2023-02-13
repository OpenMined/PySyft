.. _have_prerequisites:

==================================
I have all the dependencies
==================================

.. toctree::
   :maxdepth: 3


1. **Create a new env specifying the Python version (we recommend Python 3.8/3.9) in the terminal:**

  .. code-block:: bash

      conda create -n syft_env python=3.9
      conda activate syft_env

2. **Install PySyft and Hagrid**

To install the OpenMined stack that you need in order to deploy a node, please run:

.. code-block:: bash

   pip install -U syft hagrid


PySyft is a library which contains the tools to run privacy preserving machine learning.
Hagrid is a commandline tool that speeds up the deployment of PyGrid, the provider of a peer-to-peer network of
data owners and data scientists who can collectively train AI model using Syft.

3. **Launch the Doman Node**

You only have one final step remaining now, before you unleash the power of Hagrid!
The final step is to launch a domain node, which is as easy as:

.. code-block:: bash

   hagrid launch <name_of_domain>

To stop the running domain, run:

.. code-block:: bash

   hagrid land <name_of_domain>

But before stopping it, you can go to ``localhost:8081`` in your `browser <localhost:8081>`_ to actually interact with the PyGrid Admin UI, where you can manage as a Data Owner your datasets, as well as incoming requests from data scientist.
You can log in using the following credentials:

.. code-block:: python

   info@openmined.org

   
   changethis

Now you're all set up to fully start using PySyft!
