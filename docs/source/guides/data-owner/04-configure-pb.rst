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

Step 1: Introduction to Differential Privacy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In this step, lets understand the concept behind differential privacy and privacy budget by considering a simple scenario.

A. Scenario
##############
Consider there are ``500`` patients represented in ``2`` different datasets. One dataset is 
about general ``medical history``; the other has some but not all of the ``500`` patients 
and is focused on patients who have had ``mammography`` images taken in the past year. Now 
let's say that ``Jane Doe`` is a patient in both and is open to being studied for 
``breast cancer research`` as long as she can remain unidentifiable in the study.

B. Quick Definition: Differential Privacy
############################################
A core feature of Syft is that Syft allows you to use a ``PET(Privacy Enhancing technology)`` called 
Differential Privacy to protect the ``Privacy`` of the individuals or data subjects 
within your datasets. In this case, Differential Privacy is maintained when a 
query across both datasets ``with`` Jane Doe in it versus that same query on both 
datasets ``without`` Jane Doe creates the ``same output``. Noise is added to help average 
out and make up the difference between having Jane there versus not. 

From a top-level view, this means a couple of things:

* Differential Privacy can help a Data Scientist see trends in data ``without`` being able to ``identify`` the participants.
* The more a specific data subject involved in the query ``stands out`` in a dataset, the more noise has to be added to ``obfuscate`` them.
* There is a natural ``tradeoff`` between how much ``Privacy`` is preserved versus how much ``Accuracy`` is given.
* Data scientists can ``create answers`` from the list of allowed questions the Data Owner limited them to. This is enabled by the use and combination of different types of ``PETs``. (see the image 👇 for reference)
* Data scientists can download their allowed answers, creating a streamlined flow where ``answering questions`` using an org's Domain Server will be as easy as going to the organization's public website. (see the image 👇 for reference)

|04-configure-pb-02|

C. Quick Definition: Epsilon or Privacy Budget
################################################
Differential Privacy in practice is an algorithm that obscures an individual data subject's 
contributions to the given ``results`` of a ``query``. Privacy Budget measured in units of ``Epsilon`` 
is a way to measure the potential ``privacy loss`` or ``visibility`` you are allowing into any one of those data subjects.

.. note::
   Syft specifically ``tracks`` privacy budgets against individual data subjects instead 
   of the ``dataset`` as a whole. This may be different from other tools that use 
   Differential Privacy. This allows more ``utility`` on the dataset. 

D. Takeaway
###############
When you assign a ``privacy budget`` in Syft, you specify a ``risk tolerance`` on what 
level of ``visibility`` you feel comfortable having that Data Scientist control your 
data subjects. You are balancing this with keeping the ``accuracy`` they get on a 
helpful level and maximizing the benefit of your dataset(s). 

Let's say, in the above scenario, you allow your ``Data Scientist`` to have ``0.5e`` to 
conduct their Breast Cancer Research. You can interpret ``e`` to mean:

* That this Data Scientist will have ``0.5x`` more ``visibility`` into any one data subject like Jane Doe
* That this Data Scientist is ``0.5x`` more likely to ``learn`` something unique about Jane Doe
* That this Data Scientist can ``learn no more than 0.5e`` on Jane Doe

.. note::
   If a query would expose more than ``0.5e`` about ``Jane Doe``, then Jane Doe would get 
   dropped from the result, and noise would be used to mitigate the difference.

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

.. |04-configure-pb-02| image:: ../../_static/personas-image/data-owner/04-configure-pb-02.gif
   :width: 95%