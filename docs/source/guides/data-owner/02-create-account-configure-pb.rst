Creating User Accounts on your Domain Server
===============================================

**Data Owner Tutorials**

☑️ 00-deploy-domain

☑️ 01-upload-data

◻️ 02-create-account👈

.. note:: 
   **TIP:** To run this tutorial interactively in Jupyter Lab on your own machine type:

:: 
   
   pip install -U hagrid
   hagrid quickstart data-owner


Domain Owners can directly ``create`` user accounts for Data Scientists to use their 
domain nodes. When the domain owner creates a new user account, by default that user 
will have the lowest level of permissions and will be assigned ``0`` Privacy Budget.

In today's tutorial we will learn how to create a user account, how to check permissions, 
and how to assign a privacy budget to that user. Then we'll touch on why setting a privacy 
budget is important later in your workflow.
 

Pre-Requisites
------------------
Before you can create user accounts on your domain, you have to first:

#. Login to your Domain Node
#. Annotate your dataset with the appropriate DP metadata
#. And Upload your dataset

.. note:: 
   The above prerequisite steps are covered in the previous tutorials :doc:`How to deploy a
   Domain Node <00-deploy-domain>` and :doc:`How to upload private data to the Domain
   Node <01-upload-data>`. Please execute those steps before implementing this tutorial.

📒 Overview of this tutorial:

#. **Defining** account credentials
#. **Assigning** privacy budget
#. **Viewing** the account you just created via the Domain Management UI

|02-create-account-configure-pb-00|

Step 1: Create a User Account
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
After you have launched and logged into your domain as an ``admin``, you can create user accounts for others to use. 

.. WARNING:: 
   In this case, we will create an account for a Data Scientist from within our own team or organization.

.. note:: 
   You should only create direct user accounts on your domain node for those who have been 
   appropriately vetted and verified by your organization. To expand research done on your 
   datasets to those not directly within or verified by your organization, you should ``connect`` 
   your ``domain`` to one or more networks so that proper verification measures have been taken. 
   You can learn more about this in our "Connect Your Domain to a Network" tutorial.

There are three different ways to create an account for the user. We will discuss them in detail below:

A. Create a Data Scientist Account from the Notebook
#######################################################

To create a Data Scientists account for someone within your team or organization, you need to tell your Domain 4 things:

#. **Name**: Name of the individual
#. **Email**: Associated email address of the individual
#. **Password**: Password they would need to login into your domain (this can be changed later when they customize their ``account settings``)
#. **Budget**: When you specify a ``budget``, you assign this account with a ``privacy budget`` of ``0``. This privacy budget, set in units of ``epsilon``, is the limiter that blocks a data scientist from knowing too much about any one data subject in your dataset.

   **Note:** In future exercises, we will explore how privacy budget limits affect data subject visibility. 
   Still, for now, we will set the ``privacy budget`` to its default of ``0``, the lowest level of permission. 
   Also, by default, the role assigned to a user is a Data Scientist.

::

   In:

   # run this cell
   data_scientist_details = domain_client.create_user(
      name="Jane Doe",
      email="jane@email.com",
      password="supersecurepassword",
      budget=0
   )

   Out: 

   User created successfully!

Once you have created an account, you can ``verify`` if the user account was made successfully.

::

   In:

   # list the users that have registered to the domain
   domain_client.users

B. Users Signup to Domain to Create a Data Scientist Account via Domain URL
#################################################################################

A user can also ``signup`` or create an account on a Domain node if they have access to the ``URL`` to the Domain. 
Instead of creating an account individually for each Data Scientist, a Data Owner can ``share`` the URL to their 
Domain node and ask their team members to ``register`` to the Domain. 

To register to a Domain, you need the following details:

#. **Name**: Name of the individual
#. **Email**: Email of the individual that will be used to log into the Domain
#. **Password**: A secured password to log into the Domain
#. **Url**: Url to the domain node.
#. **Port**: Port number

::

   In:

   # run this cell
   import syft as sy
   domain_client = sy.register(
      name=”Jane Doe”,
      email=”jane@email.com”,
      password=”supersecurepassword”,
      url=”localhost”,
      port=8081
   )

On successful registration, the user is auto-logged into the domain. 

.. note:: 
   By default the role assigned to the registered user is of a ``Data Scientist`` and the assigned ``privacy budget`` is ``0``. 
   A Data Owner can further manage the registered users from the UI as indicated in Step 2.

C. Create a Data Scientist Account in PyGrid's UI
#########################################################

PyGrid's UI is meant to help Domain Owners get a bigger picture view of their domains and manage them. 

When we use the ``hagrid launch`` command to start our private data server, we define the ``port`` where 
we want to launch the server. By default, the port is launched at ``8081``.

   **Note:** Make sure your docker application is up and running in the background.

We will use this ``port number`` to visit the following UI interface at the URL:

:: 
   
   http://localhost:<port_number>
   
   e.g.
   
   http://localhost:8081


Once you are on PyGrid's web page, execute following steps to create an account for Data Scientist:

#. Login using your admin credentials
#. Create a new user account by clicking on the ``+ Create User`` button
#. Specify the following fields
	* **Name**: Name of the individual
	* **Email**: Email of the individual that will be used to log into the Domain
	* **Password**: A secured password to log into the Domain
	* **Role**: Assign them the role of Data Scientist (By default user account will take the role with the lowest amount of permission which in this case is the **Data Scientist** role.)
#. Set appropriate Privacy Budget (By default, they have ``0e`` privacy budget)

|02-create-account-configure-pb-04|

Step 2: Assign Privacy Budget
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In our specific setup, the privacy budget is measured against data subjects, not datasets. 
Therefore, the ``epsilon-ε`` value indicates how much can be learned from any data subject.

.. note:: 
   Consider there are 500 patients and 10 data scientists. This means there are 
   5000 ``epsilons`` measuring the epsilon relationships between each patient and each 
   data scientist, and our ``privacy budget`` simply says that a data scientist can’t 
   learn more than ``x`` amount of epsilon about any particular medical patient in the data.

When we use the ``hagrid launch`` command to start our private data server, we define the ``port`` where we want to 
launch the server. By default, the port is launched at ``8081``. 

|02-create-account-configure-pb-01|

We will use this port number to visit the following ``UI`` interface at the ``URL``:

::

   http://localhost:<port_number>

   e.g.

   http://localhost:8081


|02-create-account-configure-pb-02|

The default email and password for the domain are:

* **email:** info@openmined.org
* **password:** changethis

Once we're logged in, we will have the following view:

|02-create-account-configure-pb-03|

From the UI, we can ``view`` and ``control`` the following:

* **Users:** Shows a list of users that are signed to the domain. We can create, edit or delete a user from this interface.
* **Permissions:** This is a list of the different sets of roles a user can have. Each role has a set of permissions that the DO (Data Owner) can modify as per their norms.
* **Requests:** This list two types of requests Data Requests and Privacy Budget Upgrade requests.
   
   * **Data Requests:** If users want complete access to a data/variable, they can request so from the DO. Such requests will be listed here, and the DO can manually decide which ones to approve or reject.
   * **Privacy Budget Requests:** These requests pertain to the Privacy budget upgrade requested by a DS. The DO can decide if they want to assign the given privacy budget to the user or deny their requests.


Step 3: Submit Credentials
~~~~~~~~~~~~~~~~~~~~~~~~~~
Lastly, we will submit the credentials of the newly created user account to our ``domain node``. 

:: 

   In: 

   # run this cell then copy the output
   submit_credentials(data_scientist_details)

   print("Please give these details to the Data Scientist 👇🏽")
   print(data_scientist_details)

   Out:

   Data Scientist credentials successfully submitted.
   Please give these details to the Data Scientist 👇🏽
   {'name': 'ABC', 'email': 'abc@xyz.net', 'password': 'changethis', 'url': '20.253.155.183'}

You can give these details to Data Scientists so they can ``finish`` setting up their account, 
which can involve changing email and password if necessary. 

Now our domain node is available for the data scientists to use 👏
------------------------------------------------------------------

.. |02-create-account-configure-pb-00| image:: ../../_static/personas-image/data-owner/02-create-account-configure-pb-00.jpg
  :width: 95%

.. |02-create-account-configure-pb-01| image:: ../../_static/personas-image/data-owner/02-create-account-configure-pb-01.png
  :width: 95%

.. |02-create-account-configure-pb-02| image:: ../../_static/personas-image/data-owner/02-create-account-configure-pb-02.png
  :width: 95%

.. |02-create-account-configure-pb-03| image:: ../../_static/personas-image/data-owner/02-create-account-configure-pb-03.png
  :width: 95%

.. |02-create-account-configure-pb-04| image:: ../../_static/personas-image/data-owner/02-create-account-configure-pb-04.gif
  :width: 95%