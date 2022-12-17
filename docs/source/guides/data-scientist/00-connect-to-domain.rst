Connecting to a Domain Server
====================================

**Data Scientist Tutorials**

◻️ 00-connect-to-domain👈

◻️ 01-search-for-datasets
 
.. note:: 
   **TIP:** To run this tutorial interactively in Jupyter Lab on your own machine type:

:: 
   
   pip install -U hagrid
   hagrid quickstart data-scientist


Data Scientists are end users who want to perform ``computations`` or ``answer`` specific questions using 
the dataset(s) of one or more data owners. The very first thing Data Scientists have to do in order 
to submit their requests is ``login`` and ``connect`` to the Domain Server that hosts the data they would 
like to make requests off of or to connect to a network by which they can search for different 
datasets. Today's tutorial will show you how you as a Data Scientist can connect to an 
organization's domain server using PySyft.  

For connecting to a Domain Server, we will use the login credentials assigned to us by 
the Domain Owner. By default, we as Data Scientists have the lowest level of ``permission`` 
to access the data (which means data is highly private) and will be assigned a Privacy Budget of ``0``.

.. note::
   Check out this tutorial to understand how Domain Owners 
   can :doc:`create a user account <../data-owner/02-create-account-configure-pb>` on their Domain Servers.

   Throughout the tutorials, we also mean Data Scientists
   whenever we refer to users. Both are used interchangeably.

Steps to Connect to a Domain Server
-------------------------------------

📒 Overview of this tutorial:  

#. **Obtain** Login Credentials
#. **Login** to the Domain as a Data Scientist
#. **Explore** some useful starting commands


.. note::
      PyGrid Admin (the UI) is only meant to be used by domain or data owners so a data scientist 
      would never login to the domain node via the UI.

.. _step-ds-1:

Step 1: Obtain Login Credentials
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To utilize the ``privacy-enhancing`` features and play around with your ``privacy budget``, as a 
Data Scientist you must first get your login ``credentials`` from the domain owner. 
What you will need to login to the domain server is the following information:

* email
* password
* URL of the domain
* port of the domain

.. WARNING::
   Change the default username and password below to a more secure and private combination of your preference.

::

   In:

   # run this cell
   import syft as sy
   domain_client = sy.register(
      name="Alice",
      email="alice@email.com",
      password="supersecurepassword",
      url="localhost",
      port=8081
   )

.. note::
   By default, the role assigned to the registered user is of a Data Scientist, and the assigned privacy budget is 0.


Step 2: Login to the Domain as a Data Scientist
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once you have the above information you can open a ``Jupyter Notebook`` and begin ``logging`` into the domain server.

To start you will need to install syft

::

   In:

   import syft as sy

Then you can provide your login credentials by typing:

::
   
   In:

   domain = sy.login(email="____", password="____", url="____",port=8081)


Step 3: Explore some useful starting commands
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
As a Data Scientist, you can ``explore`` the Domain Server using the Python ``Syft`` library. 

.. note::
   We will explore more about each command in the next series of tutorials.

::

   In:

   # name of the domain
   domain.name

   # View datasets on the domain
   domain.datasets

   # View store on the domain
   domain.store

Awesome 👏 You have now successfully connected to a Domain Node !! 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

What's Next? 
---------------
Alright, now that you are connected to a Domain node, we would first like to look for the 
available datasets on the public network which users can join. 

   The following tutorial will show how Data Scientists can search for a dataset on the Domain Node. 
