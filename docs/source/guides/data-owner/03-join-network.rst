Joining a Network
===============================================

**Data Owner Tutorials**

☑️ 00-deploy-domain

☑️ 01-upload-data

☑️ 02-create-account

◻️ 03-join-network👈

.. note:: 
   **TIP:** To run this tutorial interactively in Jupyter Lab on your own machine type:

:: 
   
   pip install -U hagrid
   hagrid quickstart data-owner


A Network Node is a node that connects different domains to a broader base of data scientists (also known as a network's members). It is a server which exists outside of any data owner's institution, providing search & discovery, VPN, and authentication services to the network of data owners and data scientists.

.. note::
   Data is only stored on the separate Domain Servers. Network Nodes do not contain data, they simply provide an extra layer of services to Domain Nodes and Data Science users.

Let us give an example: assume you are in a hospital and the hospital has different cancer-related datasets hosted on their domain. The hospital now wants to increase the research impact their datasets can have but does not want to do so at the cost of risking a privacy leak nor at the risk of moving their data. By joining a network (for example one hosted by WHO) a Domain Owner can increase the searchability of their datasets to appropriate audiences without those datasets needing to leave the Domain servers.

In today's tutorial we will learn how to join a network and apply our domain to it.
 

🚨 Pre-Requisites Steps
---------------------------

Before you can create user accounts on your domain, you have to first:

* `Login to your Domain Node`

.. note:: 
   The above prerequisite step is covered in an existing tutorial `How to deploy a Domain Node <https://openmined.github.io/PySyft/guides/data-owner/00-deploy-domain.html>`_. Please execute those steps before implementing this tutorial.

📒 Overview of this tutorial
--------------------------------

#. **Login** to your Domain Server
#. **Finding** a Network
#. **Applying** our Domain to the Network
#. **Verifying** our Domain on the Network

Step 1: Import Syft
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Syft is the main library our Domain servers run off of, so to start we will need to import Syft so that our methods in later steps will work.
::

   In:

   # run this cell
   import syft as sy


Step 2: Login to your domain
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Once you have imported syft, and have your domain node up along with it's credentials available to you, connect and login to the domain hosted at the URL generated on the Step 4 of the Deploy Domain notebook.

.. WARNING:: 
   The below cell has default credentials, please change accordingly.

::

   In:

   # run this cell
   domain_client = sy.login(
    url="http://localhost:8081/", email="info@openmined.org", password="changethis"
   )

Step 3: Fetch all Available Networks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Now we’ve come to the main part, let’s take a look at what networks are available for us to join.
The command below will fetch all of the currently available networks, this list may change as more networks get created or as they go on and offline.

::

   In:

   # run this cell
   sy.networks

You can now choose the network that suits best your needs. After looking at the available networks, let’s choose a network that best fits our domain. For this tutorial we are going to choose the **OpenMined** network.

Step 4: Connect to the Network
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In future iterations of PyGrid Network nodes will be able to have domains join as Members or as Guests, but in today’s current iteration of PyGrid all domains start out by joining as Guests. To apply to a network as a guest we first need to connect to the network server.

Connecting to a network can be done via it's name/URL/index in the above list.

::

   In:

   # run this cell
   network_client = sy.networks[0]

On successful login, the `network_client` will contain an authenticated client to the network.

Step 5: Fetch all Domains on the Network
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Now that we have an authenticated client with the network, let's fetch and see the currently connected domains on the network.

We can list all of them with the below command:

::

   In:

   # run this cell
   network_client.domains

Since we have not applied our domain yet, it should not be visible on the output of the above command.

Step 6: Apply our Domain to the Network
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In this step, we will be joining the OpenMined network. If our application to join gets accepted, our domain will then be listed among the available domains on this network which will help Data Scientists find and work from our datasets.

.. note::
   This step might have multiple retries before actually getting connected, so please don’t worry!

The below command will apply our domain node to the network we just authenticated with

::

   In:

   # run this cell
   domain_client.apply_to_network(network_client)


Step 7: Verify our Domain on the same Network
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In this step, we will verify whether we have successfully joined the network node or not. We will do this by listing the domains available on this network and seeing whether our domain appears.

::

   In:

   # run this cell
   network_client.domains

If you can see your domain's name here, then hoorah!

If you haven't, don’t worry, go through the above steps and see if you missed anything.

Step 8: Verify the VPN status
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Now, let us verify that our domain is succesfully connected to the Network node via VPN.

Run the cell below as mentioned:

::

   In:

   # run this cell
   domain_client.vpn_status()

You should receive the domain ID in the `peers list` in the connected field. This confirms our connection to the network, Yay!

Now our domain node applied on the network and we have succesfully joined it!👏