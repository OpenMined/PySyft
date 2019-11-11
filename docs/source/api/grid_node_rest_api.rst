Grid Node Rest API
==================

All HTTP/HTTPS endpoints will be detailed in this document.

Upload Models
-------------

| **URL** : ``/serve-model``
| **Description** : Used to save a model (jit/plan).
| **Method** : ``POST``
| **Content-Type** : multipart/form-data \|\| application/json
| **Auth required** : NO (can be changed)

Request Body:
^^^^^^^^^^^^^

.. code:: json

    {       
            "encoding": "encode_type",
            "model_id": "modeid",
            "allow_download": "Boolean",
            "allow_remote_inference": "Boolean",
            "model": "serialized_and_decoded_model"
    }

Status Code: 200 OK
^^^^^^^^^^^^^^^^^^^

.. code:: json

    {
        "success": "Boolean",
        "message": "Model saved with id: <model_id>"
    }

Status Code: 409 Conflict
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: json

    {
            "error":  "Model with id: <model_id> already eixsts."
    }

Delete Model
------------

| **URL** : ``/delete-model``
| **Description** : Used to delete models by their ids (jit/plan).
| **Method** : ``POST``
| **Content-Type** : application/json
| **Auth required** : NO (can be changed) #### Request Body

.. code:: json

    {
        "model_id" : "model id"
    }

Status Code: 200 OK
^^^^^^^^^^^^^^^^^^^

.. code:: json

    {
        "success": "Boolean",
        "message": "Model deleted with success!"
    }

Status Code: 404 Not Found
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: json

    {
        "error" : "Model not found!" 
    }

List registered models
----------------------

| **URL** : ``/models``
| **Description**: Used to get a list of stored models, return model's
ids.
| **Method** : ``GET``
| **Content-Type** : application/json
| **Auth required** : NO (can be changed)

Status Code: 200 OK
'''''''''''''''''''

.. code:: json

    {
        "success": "Boolean",
        "models": [ "modelid1", "modelid2" ] 
    }

Check Models permissions
------------------------

| **URL** : ``/is_model_copy_allowed/<model_id>``
| **Description**: Used to check if some model can be copied/downloaded.
| **Method** : ``GET``
| **Content-Type** : application/json
| **Auth required** : NO (can be changed)

Status Code: 200 OK
'''''''''''''''''''

.. code:: json

    {   
            "success": "True"
    }

Status Code: 200 OK
'''''''''''''''''''

.. code:: json

    {
        "success": "False",
        "error": "You're not allowed to download this model."
    }

Download Model
--------------

| **URL** : ``/get_model/<model_id>``
| **Description** : Used to download a specific model.
| **Method** : ``GET``
| **Auth required** : NO (can be changed)

Status Code: 200 OK
'''''''''''''''''''

.. code:: json

    {       
            "serialized_model":  "serialized_and_decoded_model"
    }

Status Code: 404 Not Found
''''''''''''''''''''''''''

.. code:: json

    {       
            "error":  "Model not found."
    }

Status Code: 403 Forbidden
''''''''''''''''''''''''''

.. code:: json

    {       
            "error":  "You're not allowed to download this model."
    }

Run Data Inferences
-------------------

| **URL** : ``/model/<model_id>``
| **Description** : Used to run data inferences at some model. Returns
inference's results.
| **Method** : ``POST``
| **Content-Type** : application/json
| **Auth required** : NO (can be changed)

Request Body:
^^^^^^^^^^^^^

.. code:: json

    {       
            "encoding": "encode_type",
            "data":  "serialized_and_decoded_data"
    }

Status Code: 200 OK
^^^^^^^^^^^^^^^^^^^

.. code:: json

    {       
            "status":  "Boolean",
            "prediction": "serialized_and_decoded_inferece"
    }

Status Code: 403 Forbidden
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: json

    {       
            "status":  "Boolean",
            "error" : "You're not allowed to run inference on this model."
    }

Search SMPC Models
------------------

| **URL** : ``/search-encrypted-models``
| **Description** : Search smpc model by id, if found, returns worker's
addresses.
| **Method** : ``POST``
| **Content-Type** : application/json
| **Auth required** : NO (can be changed)

Request Body
^^^^^^^^^^^^

.. code:: json

    {
        "model_id" : "model_id"
    }

Status Code: 200 OK
^^^^^^^^^^^^^^^^^^^

.. code:: json

    { 
        "workers" : [ "http://worker1address.com", "http://worker2address.com" ],
        "crypto_provider" : [ "http://cryptoprovideraddress.com" ]
    }

Status Code: 400 Bad Request
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: json

    { 
        "error": "Invalid payload format"
    }

Status Code: 404 Not Found
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: json

    { 
        "error": "Model ID not found!"
    }

Dataset Tags
------------

| **URL** : ``/dataset-tags``
| **Description** : Get all dataset tags stored in this node.
| **Method** : ``GET``
| **Content-Type** : application/json
| **Auth required** : NO (can be changed)

Status Code: 200 OK
^^^^^^^^^^^^^^^^^^^

.. code:: json

    [ "dataset-tag1", "dataset-tag2", "dataset-tag3"]

Search dataset / tensor
-----------------------

| **URL** : ``/search``
| **Description** : Check if grid node have the desired dataset tags.
| **Method** : ``POST``
| **Content-Type** : application/json
| **Auth required** : NO (can be changed)

Request Body:
^^^^^^^^^^^^^

.. code:: json

    {
        "query": ["#tag1", "#tag2", "#tag3"]
    }

Status Code : 400 Bad Request
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: json

    {
        "error": "Invalid payload format"
    }

Status Code: 200 OK
^^^^^^^^^^^^^^^^^^^

.. code:: json

    {
        "content": "Boolean"
    }

