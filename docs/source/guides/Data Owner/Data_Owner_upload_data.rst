Data Owner - How to Upload Private Data to the Domain Server
============================================================

Welcome back to another ``Data Owner tutorial``. In the last tutorial,
you learned how Data Owners can deploy the domain node that represents
the theoretical organization’s ``private data servers``. But right now,
the node you just deployed is empty.

After today’s tutorial, you will learn how to ``upload`` the
``private data``, which involves annotating and doing ETL before
uploading it to our Domain Node/server.

   **Note:** Throughout the tutorials, we also mean Domain Servers
   whenever we refer to Domain Node. Both mean the same and are used
   interchangeably.

Step to Upload Private Data
---------------------------

The steps covered in this tutorial include: 
#. **preprocess** of data 
#. mark it with the correct **metadata** 
#. **upload** it to Domain node

|Data_Owner_upload_data01|

   **Note:** For the ease of running all the steps shown in this
   tutorial, we prefer using the below command.

::


   hagrid quickstart https://github.com/OpenMined/PySyft/tree/dev/notebooks/quickstart/Tutorial_Notebooks/Data_Owner_upload_data.ipynb

Step 1: Import Syft
~~~~~~~~~~~~~~~~~~~

The first step is to ``configure`` Privacy Enhancing Technologies
(PETs). For this, you need OpenMined’s Syft library.

Lets import Syft by running the below cell:

::

   In:

   # run this cell
   import syft as sy
   from utils import *
   print("Syft is imported")

   Out: Syft is imported

Step 2: Python Client Login
~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is important to note that only the Domain node ``administrator`` can
upload data. So before the Domain node lets you upload private data, you
must prove you are an admin by ``logging`` in.

In this case, you have to give some default credentials like: \* IP
Address of the host \* Email and password

   **WARNING:** CHANGE YOUR USERNAME AND PASSWORD!!!

::

   In:

   domain_client = sy.login(
       url="20.31.143.254",
       email="info@openmined.org",
       password="changethis"
   )

   Out:

   Connecting to 20.253.155.183... done! Logging into openmined... done!

Lovely :) You have just logged in to your Domain.

Step 3: Get Dataset
~~~~~~~~~~~~~~~~~~~

For this tutorial, we will use the simple ``age`` data of a family of 4
members.

   **IMPORTANT:** In real-world applications, the dataset is broken into
   subsets and dispersed among participants in the event of Remote Data
   Science.

::

   In:

   # import pandas
   import pandas as pd

   data = {'ID': ['011', '015', '022', '034'],
           'Age': [40, 39, 9, 8]}

   dataset = pd.DataFrame(data)

   Out:

   ID  Age
   011   40
   015   39
   022    9
   034    8

Step 4: Annotate Data for Automatic DP
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the next step, we annotate our data with some Auto DP
``(Automatic Differential Privacy)`` metadata using PETs. This allows
the syft library to ``protect`` and ``adjust`` the visibility different
data scientists have into any one of the data subjects.

Important steps:
^^^^^^^^^^^^^^^^

-  ``data subjects`` are entities whose privacy we want to protect
-  each feature needs to define the appropriate ``minimum`` and
   ``maximum`` ranges
-  when defining min and max values, we are actually defining the
   ``theoretical`` amount of values that could be learned about that
   aspect.
-  in our case, the minimum age can be ``0``; theoretically, the maximum
   age can be ``115`` or the oldest living person to date.

::

   In: 

   # run this cell
   data_subjects = DataSubjectList.from_series(dataset["ID"])

   age_data = sy.Tensor(dataset["Age"]).annotated_with_dp_metadata(
       min_val=0, max_val=100, data_subjects=data_subjects
   )

..

   **Note:** If your project has a training set, validation set and test
   set, you must annotate each data set with Auto DP metadata.

Step 5: Upload & Check the Dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once you have prepared your data, it’s time to upload it to the Domain
node. To help Data Scientists later ``search`` and ``discover`` our
datasets, we will add details like a ``name`` and a ``description`` of
what this dataset represents.

   **Note:** If your project has a train, validation and test set, you
   need to add them as assets. In this case, it is just our age column.

::

   In:

   # run this cell
   domain_client.load_dataset(
       name="Family_Age_Dataset",
       assets={
           "Age_Data": age_data,
       },
       description="Our data set contains Age of Family of 4 members with \ 
       their unique ID's. There are two columns and 4 rows in our dataset."
   )

   Out: 

   Dataset is uploaded successfully !!!

To ``check`` the dataset you uploaded to the Domain Node, go ahead and
run the below command, and it will list ``all`` the datasets on this
Domain with their Names, Descriptions, Assets, and Unique IDs.

::

   In:

   # run this cell
   domain_client.datasets

Awesome 👏 !! You have uploaded the dataset onto your Domain node.
-----------------------------------------------------------------

By uploading the dataset onto the Domain Node, Data Owners are opening
up the possibilities of different Data Scientists being able to study it
without downloading it and without the Data Owners doing any
experiment-specific work while Data Scientists are studying their
private data.

What’s Next? 
------------
Alright, so we have walked through **“How to deploy a
Domain Node”** and **“How to prepare and upload a dataset to that Domain
Node”** so that Data Scientists can study our datasets without being
able to download them.

   In the following tutorial, we will see how Data Scientists can find
   datasets and work across all the different Domain nodes.

.. |Data_Owner_upload_data01| image:: ../../_static/personas_image/DataOwner/Data_Owner_upload_data01.jpg
  :width: 95%
  :alt: Data Owner-Upload Private Data