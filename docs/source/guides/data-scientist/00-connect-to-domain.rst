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


Data Scientists are end users who desire to perform ``computations`` or ``answer`` a specific question 
using one or more data owners' datasets. The very first thing Data Scientists have to do is ``login`` 
and ``connect`` to the Domain Server of their organization's private data server. Today's tutorial shows 
how Data Scientists can connect to an organization's private data servers using ``PyGrids UI``.  

For connecting to a Domain Server, we will use PyGrid's UI interface with the login credentials 
assigned by the Domain Owners. By default, the Data Scientists have the lowest level of ``permission`` 
to access the data (which means data is highly private) and will be assigned a Privacy Budget of ``0``.

.. note::
   Check out this tutorial to understand how Domain Owners 
   can :doc:`create a user account <../data-owner/02-create-account-configure-pb>` on their Domain Servers.

   Throughout the tutorials, we also mean Data Scientists
   whenever we refer to users. Both are used interchangeably.

Steps to Connect to a Domain Server
-------------------------------------

📒 Overview of this tutorial:  

#. **Obtain** the Login Credentials
#. **Start** a private Data Server
#. **Login** to your Server
#. **Explore** the User HomePage


Step 1: Obtain the Login Credentials
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To utilize the ``privacy-enhancing`` features and play around with different ``privacy budgets``, 
Data Scientists must first get their login ``credentials``, using which they can login into their 
organization's private data servers. Here are a few points to follow to get your login credentials:

* If your account was created by the Domain Owners within your organization, you could ask them to 
  securely ``share`` the credentials with you. If you do not have a domain owner, you can ``create`` 
  one locally following the tutorials starting here: `data-owner/00-deploy-domain <../data-owner/00-deploy-domain.html>`_.
  
* You can also signup or create an account on a Domain node if you have access to the ``URL`` to the Domain. 
  To ``register`` yourself to the Domain, you need to run the following code:

.. WARNING::
   Change the default username and password below to a more secure and private combination of your preference.

::

   '''
   Name: Name of the Data Scientist
   Email: Email of the Data Scientist
   Password: A secured password to log into the Domain
   Url: Url to the domain node.
   Port: Port number
   '''

   In:

   # run this cell
   import syft as sy
   domain_client = sy.register(
      name="Jane Doe",
      email="jane@email.com",
      password="supersecurepassword",
      url="localhost",
      port=8081
   )

.. note::
   By default, the role assigned to the registered user is of a Data Scientist, and the assigned privacy budget is 0.


Step 2: Start a Private Data Server
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
PyGrid's UI is meant to help users get a bigger picture view of their organization's domains, analyze 
the ``datasets``, manage ``permissions``, and play with different ``privacy budgets``. 

We will use the ``hagrid launch`` command to start a private data server. By default, the port is launched at ``8081``.

.. note::
   Make sure your docker application is up and running in the background.

We will use this port number to visit the following UI interface at the ``URL``:

::

   http://localhost:<port_number>

   e.g.

   http://localhost:8081

Once the ``hagrid launch`` command is executed successfully, you will see the message as shown in the image below:

|00-connect-to-domain-00|
   

Step 3: Login to your Server
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~















.. |00-connect-to-domain-00| image:: ../../_static/personas-image/data-scientist/00-connect-to-domain-00.png
   :width: 95%