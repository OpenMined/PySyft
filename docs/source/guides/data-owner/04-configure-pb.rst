Configuring Privacy Budget on your Domain Server
==================================================

**Data Owner Tutorials**

☑️ 00-deploy-domain

☑️ 01-upload-data

☑️ 02-create-account

☑️ 03-join-network

◻️ 04-configure-pb👈

.. note:: 
   **TIP:** To run this tutorial interactively in Jupyter Lab on your own machine type:

:: 
   
   pip install -U hagrid
   hagrid quickstart data-owner


A privacy budget is a collection of quantitative measures through which a Data Owner can 
pre-determine the degree of information access they grant to a user using their domain server. 
As we saw in the :doc:`creating user accounts tutorial <02-create-account-configure-pb>`, when you 
create a user account, that user is assigned the ``lowest level`` of permissions to access that 
data, which means the data is highly private. We assigned a user the role of ``Data Scientist`` 
and a privacy budget of ``0`` by default. 

In today's tutorial, you will discover the underlying concept behind privacy budget 
(Differential Privacy), epsilon value or privacy budget, and how configuring your privacy 
budget determines how much can be learned from any data subject.
 

🚨 Pre-Requisites Steps
---------------------------

Before you can create user accounts on your domain, you have to first:

#. :ref:`Login to your Domain Node <step2>`
#. :ref:`Annotate your dataset with the appropriate DP metadata <step4>`
#. :ref:`Upload your dataset to Domain <step5>`
#. :ref:`Create a user account <step6>`

.. note:: 
   The above prerequisite steps are covered in the previous tutorial 
   :doc:`How to upload private data to the Domain Node <01-upload-data>` and
   :doc:`Creating User Accounts on your Domain Server <02-create-account-configure-pb>`. Please execute those 
   steps before implementing this tutorial.

📒 Overview of this tutorial
---------------------------

#. **Introduction** to Differential Privacy
#. **Login** to PyGrid UI as a Domain Admin
#. **Explore** different Privacy Budget

Step 2: Login to PyGrid UI as a Domain Admin
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
When we use the ``hagrid launch`` command to start our private data server, we define 
the ``port`` where we want to launch the server.

.. note:: 
   By default, the port is launched at ``8081``.

|04-configure-pb-00|

We will use this port number to visit the following UI interface at the ``URL``:

::

   http://localhost:<port_number>

   e.g.

   http://localhost:8081

|04-configure-pb-01|

The default email and password for the domain are:

* **email:** info@openmined.org
* **password:** changethis

Once we're logged in, you can move to the next section, which explores setting a privacy budget.



------------------------------------------------------------------

.. |04-configure-pb-00| image:: ../../_static/personas-image/data-owner/04-configure-pb-00.png
   :width: 95%

.. |04-configure-pb-01| image:: ../../_static/personas-image/data-owner/04-configure-pb-01.png
   :width: 50%