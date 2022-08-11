Data Owner - Upload Private Data
================================

Welcome back to another Data Owner tutorial. In the last tutorial, you
learned how Data Owners can deploy the domain node that represents the
theoretical organization’s private data servers. But right now, the node
you just deployed is empty.

After today’s tutorial, you will learn how to upload the private data,
which involves annotating and doing ETL before uploading it to our
Domain Node/server.

   **Note:** Throughout the tutorials, we also mean Domain Servers
   whenever we refer to Domain Node. Both mean the same and are used
   interchangeably.

Uploading Private Data
----------------------

The steps covered in this tutorial include: 

#. preprocessing of the data 
#. marking it with the correct metadata 
#. uploading it to the Domain node

|Data_Owner_upload_data01|

   **Note:** For the ease of running all the steps shown in this tutorial, we
   prefer using the below command.

::

   hagrid quickstart https://github.com/OpenMined/PySyft/tree/dev/notebooks/Tutorial_Notebooks/Data_Owner_upload_data.ipynb

Step1: Import Syft
~~~~~~~~~~~~~~~~~~

The first step is to configure Privacy Enhancing Technologies (PETs).
For this, you need OpenMined’s Syft library.

Lets import Syft by running the below cell:

::

   In:

   # run this cell
   import syft as sy
   from utils import *
   print("Syft is imported")

   Out: Syft is imported

Step2: Python Client Login
~~~~~~~~~~~~~~~~~~~~~~~~~~

It is important to note that only the Domain node administrator can
upload data. So before the Domain node lets you upload private data, you
must prove you are an admin by logging in.

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

Step3: Get Dataset
~~~~~~~~~~~~~~~~~~

For this tutorial, we will use the Brest Histopathology Images dataset
from
`Kaggle <https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images>`__.
This dataset focuses on Invasive Ductal Carcinoma (IDC), the most common
subtype of all breast cancers.

   **IMPORTANT THING TO KNOW ABOUT THE DATASET:** The dataset itself is
   broken into subsets and dispersed between each of the participants in
   the event of Remote Data Science. This way, we can genuinely mimic
   when we have our Data Scientists account what it would be like to
   pull a complete image of understanding together.

::

   In:

   # edit MY_DATASET_URL then run this cell

   MY_DATASET_URL = ""

   dataset = download_dataset(MY_DATASET_URL)

Once this cell is finished running, you can see a couple of fancy images
from the dataset you just downloaded.

Step4: Preview Dataset
~~~~~~~~~~~~~~~~~~~~~~

You can get a preview of the dataset using the ``head`` method. It
prints the first 5 rows along with the columns and column labels of the
data.

::

   In:

   dataset.head()

Step5: Preprocess Data
~~~~~~~~~~~~~~~~~~~~~~

The next step involves preparing the dataset for any potential Data
Scientist. So we go ahead and preprocess the data and split the dataset
into different training, validation and testing sets.

::

   In:

   # run this cell to split the data
   train, val, test = split_and_preprocess_dataset(data=dataset)

   Out:

   Splitting dataset into train, validation, and test sets.
   Preprocessing the dataset...
   Preprocessing completed.


.. |Data_Owner_upload_data01| image:: ../../_static/personas_image/DataOwner/Data_Owner_upload_data01.jpg
  :width: 95%
  :alt: Data Owner-Upload Private Data