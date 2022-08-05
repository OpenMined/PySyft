Data Owner - Deploy a Domain Server
===============================================

Data owners are important ``remote data science`` members with
``datasets`` 💾 they want to make available for
``study by an outside party``.

Today’s tutorial will help you understand why and how Data owners can
``launch`` and ``deploy`` the ``domain node``.

   **Note:** Throughout the tutorials, we also mean Domain Servers whenever we refer to Domain Node. Both mean the same and are used interchangeably. 

Why Deploy Domain Servers?
--------------------------

The concept of Remote Data Science starts with a ``server-based model``
that we call ``Domain Server``. It allows people/data owners 👨 to load
their ``private data`` into these servers and ``create`` an account with
a username and password for ``Data Scientists`` 💻.

The ``advantage`` of using a ``Domain Server`` is that a Data Scientist
can now ``answer questions`` (on your dataset) using an organization’s
domain server as promptly as going to their public website.


|data_owner_tutorial01|


Can you notice what’s missing in the above diagram? After ``deploying``
a domain, there are no ``data partnerships``, ``meetings`` with
``lawyers``, lengthy ``phone calls``, ``background checks`` or any
``waiting`` between the organization and the data scientists.

The ``organization`` 🏫 that has deployed the domain (with private data)
retains ``governance`` 🖊 over the ``information`` they steward and never
``shares a copy`` of the ``data``.

Deploy a Domain
---------------

In a nutshell, you will be: 

* ``installing`` the required software 
* ``running`` the servers 
* ``checking`` the status of deployed server

|data_owner_tutorial011|

Few things to make a note of before starting: 

- ``PySyft`` = Privacy-Preserving Library 
- ``PyGrid`` = Networking and Management Platform 
- ``HAGrid`` = Deployment and Command Line Tool

   **Note:** For the ease of running all the steps shown in this tutorial, we
   prefer using the below command.

::

   hagrid quickstart https://github.com/OpenMined/PySyft/tree/dev/notebooks/Tutorial_Notebooks/Tutorial_01_DataOwner.ipynb

Step1: Install PiP
~~~~~~~~~~~~~~~~~~

Assuming some of you ``don't`` have all the ``packages`` of ``Python``
installed, the first step is ``installing`` the ``latest pip version``.

In your jupyter ``notebook`` cell, ``execute`` the following command:

::

   In:

   # run this cell
   ! sudo apt update && sudo apt install python3-pip

   ! echo "\n✅ Step Complete\n"

Step2: Install HAGrid
~~~~~~~~~~~~~~~~~~~~~

The next step is to ``install`` OpenMined’s ``command line tool`` called
``HAGrid``. It helps to install needed ``dependencies`` to ``launch``
and ``run`` a ``domain node`` correctly. HAGrid also allows your IT
teams to deploy different ``nodes``, continuously ``monitors`` them, and
ensures they are deployed ``correctly``.

To install it, ``run`` the below command:

::

   In:

   # run this cell
   ! pip install -U hagrid

   ! echo "\n✅ Step Complete\n"

Step3: Install Syft
~~~~~~~~~~~~~~~~~~~

One last tool to install is OpenMined’s Syft library. ``PySyft`` is an
open-source library for ``privacy-enhancing technologies`` like
``Homographic Encryption``, ``Differential Privacy``, and
``Secure Multi-party Computation``. It allows you to do ``private`` &
``secure`` deep learning and remote data science in Python.

   **Note:** Syft is under ``active development`` and is not yet ready
   for total pilots on private data without our assistance. As
   ``early access`` participants, please ``contact us`` via
   `slack <https://communityinviter.com/apps/openmined/openmined/>`__ or
   ``email`` if you would like to ask a ``question`` or have a
   ``use case`` that you would like to propose.

For Data Owners, Syft is a library that can ``support`` a Data
scientist’s workflow without that Data Scientist having a
``direct copy`` of your data. But more of it in the following tutorials.

``Run`` the below command in your notebook cell:

::

   In:

   # run this cell
   ! pip install --pre syft

   ! echo "\n✅ Step Complete\n"

..

   **Note:** The next step will show you how to launch a domain node. If
   you run into an ``issue`` installing the above tools, consider
   looking for the ``error`` you are getting on our
   `GitHub-Issue <https://github.com/OpenMined/PySyft/issues>`__ page.
   Still not able to figure out the problem, don’t worry. We are here to
   help you. Join the OpenMined
   `slack <https://communityinviter.com/apps/openmined/openmined/>`__
   community and explain your problem in the ``#general`` channel, and
   any one of us might be able to help you.

Step4: Launch Domain
~~~~~~~~~~~~~~~~~~~~

Great work, people!! Once you have installed all the dependencies, it is
time to ``use HAGrid`` to ``launch`` your ``Domain Node``.

To launch a domain node, there are ``three different things`` that you
need to know: 

1. **What type of node do you need to deploy?** There are
two different types of nodes: ``Domain Node`` and ``Network Node``. By
``default``, HAGrid launches the ``primary`` node that is our Domain
Node. 

2. **Where are you going to launch this node to?** We need to
``specify`` that we want to launch it to the ``docker container`` at
port ``80``. 

3. **What is the name of your Domain Node going to be?**
For that, don’t forget to ``specify`` the ``DOMAIN_NAME`` to your
``preference``.

You can simply ``run`` the below commands in your notebook, and a domain
node will be ``launched``.

::

   In: 

   # edit DOMAIN_NAME and run this cell

   DOMAIN_NAME = "My Institution Name"

   ! hagrid launch {DOMAIN_NAME} to docker:80 --tag=latest --tail=false

   ! echo "\n✅ Step Complete\n"

While this command runs, you will see ``various`` ``volumes`` and
``containers`` being ``created``. Once this step is complete, move on to
the ``next`` step, where we will learn to ``monitor`` the ``health`` of
our ``Domain`` ``Node``.

Step5: Check Domain
~~~~~~~~~~~~~~~~~~~

Now, let us do a quick health check to ``ensure`` the Domain Node is
``running`` and is ``healthy``.

   **Note:** One exciting ``benefit`` of HAGrid is that it makes it
   easier for your organization/ IT department to ``monitor`` &
   ``maintain`` the status of your system as you move forward with other
   steps.

::

   In:

   # run this cell
   ! hagrid check --wait --silent

   ! echo "\n✅ Step Complete\n"

   Out: 

   Detecting External IP...
   ┏━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━┓
   ┃ PyGrid    ┃ Info                        ┃    ┃
   ┡━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━┩
   │ host      │ 20.31.143.254               │ ✅ │
   │ UI (βeta) │ http://20.31.143.254/login  │ ✅ │
   │ api       │ http://20.31.143.254/api/v1 │ ✅ │
   │ ssh       │ hagrid ssh 20.31.143.254    │ ✅ │
   │ jupyter   │ http://20.31.143.254:8888   │ ✅ │
   └───────────┴─────────────────────────────┴────┘

   ✅ Step Complete

If your ``output`` is similar to the above image, ``voila!!`` A
``Domain`` ``Node`` was just ``born``. When it’s ready, you will see the
following in the ``output``:

-  **host:** ``IP address`` of the launched Domain Node.
-  **UI (Beta):** Link to an ``admin portal`` that allows you to
   ``control`` Domain Node from a web ``browser``.
-  **api:** ``Application layer`` that we run in our notebooks to make
   the experience more straightforward and intuitive.
-  **Ssh:** ``Key`` to get into virtual machine.
-  **jupyter:** ``Notebook`` ``environment`` you will use to upload your
   datasets.

Congratulations 👏 You have now successfully deployed a Domain Node.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now what?
---------

Once Data Owners have ``deployed`` the ``Domain Node`` representing
their theoretical organization’s ``private data servers``, the next step
is to ``upload private data`` for research or project use.

   In our following tutorial, we will see how Data Owners can preprocess
   the data, mark it with correct metadata and upload it to the Domain
   Node(which we just learned to deploy).

.. |data_owner_tutorial01| image:: ../../_static/personas_image/DataOwner/data_owner_tutorial01.gif
  :width: 95%
  :alt: Data Owner-Deploy Domain Node

.. |data_owner_tutorial011| image:: ../../_static/personas_image/DataOwner/data_owner_tutorial011.jpg
  :width: 95%
  :alt: Data Owner-Deploy Domain Node