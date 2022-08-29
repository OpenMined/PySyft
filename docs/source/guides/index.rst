.. _resources:

.. toctree::
   :maxdepth: 3

How-to Guides
======================

The PySyft guides are written in ``Jupyter Notebooks`` in a hosted network environment that requires no setup. Follow the instructions to get started. 

    **Note:** This section is incomplete and currently in progress.

**Important:** To simplify the ``installation`` process, we have made an ``install wizard`` notebook that 
will help you execute all the required commands needed to install the latest version of the 
dependencies like ``PiP``, ``HagRID``, and ``Syft``.

Use the below command to import the install wizard notebook into your environment:

::

   hagrid quickstart https://github.com/OpenMined/PySyft/tree/dev/notebooks/quickstart/01-install-wizard.ipynb


Once you have the installation completed, the best place to start is by ``identifying`` your role. 


#. How to use PySyft for Data Owner (in-progress)
    #. :doc:`How to Deploy a Domain Server <data-owner/00-deploy-domain>`
    #. :doc:`How to Upload Private Data to the Domain Server <data-owner/01-upload-data>`
    #. :doc:`How to create a Data Scientist account and configure your domain server with a privacy budget <data-owner/02-create-account-configure-pb>`

#. How to use PySyft for Data Scientist (coming soon)
    #. How to install Syft on your machine
    #. How can data scientists connect to a Domain server
    #. How to search for datasets on the Network server 
    #. How data scientists can train models on private data

#. How to use PySyft for Data Engineers (coming soon)
    #. How to setup development mode locally
    #. How to deploy Syft to Microsoft Azure Platform
    #. How to deploy Syft to Google Cloud Platform (GCP)
    #. How to deploy Syft to Kubernetes Cluster