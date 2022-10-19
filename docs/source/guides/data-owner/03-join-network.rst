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


A Network Node is a node that connects different domains to a broader base of data scientists (also known as a network's members). It is a server which exists outside of any data owner's institution, providing services to the network of data owners and data scientists.

In short, a Network node provides a secure interface between its cohorts or Domains and its members or Data Scientists.

Let us give an example: assume you are in a hospital and the hospital has different cancer related datasets hosted on their domain. The hospital's data owners now want to increase the visibility and searchability of these datasets, so that more and more researches and doctors can utilise these datasets and advance our understanding and diagnosis of cancer.

However, due to privacy concerns, they do not want to provide access to random actiors, such as sharing the URL of the domain with everyone. In order to tackle this privacy issue and make the dataset still accessible, the domain owner can join a Network Node (for example the one hosted by WHO) hence opening the accessibility of their datasets to a much larger audience in a private and secure manner.

In today's tutorial we will learn how to join a network and apply our domain to it.
 

🚨 Pre-Requisites Steps
---------------------------

Before you can create user accounts on your domain, you have to first:

* `Login to your Domain Node`

.. note:: 
   The above prerequisite step is covered in an existing tutorial `How to deploy a Domain Node`. Please execute those steps before implementing this tutorial.

📒 Overview of this tutorial
--------------------------------

#. **Login** to your Domain Server
#. **Finding** a Network
#. **Applying** our Domain to the Network
#. **Verifying** our Domain on the Network

Step 1: Import Syft
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We import syft to use our methods and functions in the further steps.
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
Now we come to the main part, let's fetch all the publicly visible and available networks. This is how one can discover and interact with existing Network nodes. The below command will fetch all the networks as at the time of running.

::

   In:

   # run this cell
   sy.networks

You can now choose the network that suits best your needs. We chose **OpenMined Testnet**, but you can change it with any other available network.

Step 4: Connect to the Network
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We will now connect to a network. You can connect to it with authorized credentials or as a Guest user. If you connect as a Guest user, you get limited privileges.

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
In this step, we will verify that we have succesfully joined the network node or not. We will simply do this by listing the domains on this network and we should be able to see our domain too now!

::

   In:

   # run this cell
   network_client.domains

If you can see your domain's name here, then hoorah!

If you havn't, do not worry, go through the above steps and see if you did not miss any and are following exactly the way you are supposed to!

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