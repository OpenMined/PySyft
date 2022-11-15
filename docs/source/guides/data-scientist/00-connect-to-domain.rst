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
   can :doc:`create a user account <02-create-account-configure-pb>` on their Domain Servers.


Steps to Connect to a Domain Server
-------------------------------------

📒 Overview of this tutorial:  

#. **Obtain** the Login Credentials
#. **Start** a private Data Server
#. **Login** to your Server
#. **Explore** the User HomePage


Step 1: Obtain the Login Credentials
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


   