Grid Gateway Rest API
=====================

All HTTP/HTTPS endpoints will be detailed in this document.

Register grid node
------------------

| **URL** : ``/join``
| **Description** : register new grid node into grid network.
| **Method** : ``POST``
| **Content-Type** : application/json
| **Auth required** : NO (can be changed)

Request Body:
^^^^^^^^^^^^^

.. code:: json

    {       
        "node-id" : "node_id",
        "node-address" : "http://nodeaddress.com"
    }

Status Code: 200 OK
^^^^^^^^^^^^^^^^^^^

.. code:: json

    {
        "message": "Successfully Connected!"
    }

Status Code: 400 Bad Request
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: json

    {
            "message":  "Invalid json."
    }

Status Code: 409 Conflict
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: json

    {
            "message":  "This ID has already been registered."
    }

Connected grid nodes
--------------------

| **URL** : ``/connected-nodes``
| **Description** : Return a list with connected grid nodes.
| **Method** : ``GET``
| **Auth required** : NO (can be changed)

Status Code: 200 OK
^^^^^^^^^^^^^^^^^^^

.. code:: json

    {
        "grid-nodes": ["node1", "node2", "node3"]
    }

Choose nodes to host encrypted models
-------------------------------------

| **URL** : ``/choose-encrypted-model-host``
| **Description**: Return a list of tuples of available nodes to host
encrypted model.
| **Method** : ``GET``
| **Auth required** : NO (can be changed)

Status Code: 200 OK
'''''''''''''''''''

.. code:: json

    [ ["node_id1", "node_address1"], ["node_id2", "node_address2"], ["node_id3", "node_address3"]]

Choose host to non-encrypted model
----------------------------------

| **URL** : ``/choose-model-host``
| **Description**: Return a list of tuples of available nodes to host
non-encrypted model.
| **Method** : ``GET``
| **Auth required** : NO (can be changed)

Status Code: 200 OK
'''''''''''''''''''

.. code:: json

    [ ["node_id1", "node_address"], ["node_id2", "node_address2"] ]

Search encrypted model
----------------------

| **URL** : ``/search-encrypted-model``
| **Description** : Search encrypted model on grid network.
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
'''''''''''''''''''

.. code:: json

    {       
        "<node_id>": "<node_address>",
        "nodes" : {
                "workers": [["workerid1", "workeraddress1"], ["workerid2","workeraddress2"]],
                "crypto_provider" : ["crypto_providerid", "crypto_provideraddress"]
        }
    }

Status Code: 400 Bad Request
''''''''''''''''''''''''''''

.. code:: json

    {
        "message": "Invalid json fields."
    }

Search Model
------------

| **URL** : ``/search-model``
| **Description** : Search non-encrypted model on grid network.
| **Method** : ``POST``
| **Content-Type** : application/json
| **Auth required** : NO (can be changed)

Status Code: 200 OK
'''''''''''''''''''

.. code:: json

    [["model_id1", "model_address1"], ["model_id2", "model_address2"], ["model_id3", "model_address3"]]

Status Code: 400 Bad Request
''''''''''''''''''''''''''''

.. code:: json

    {
        "message": "Invalid json fields."
    }

Status Code: 200 OK
'''''''''''''''''''

.. code:: json

    [ ["node_id1", "node_address"], ["node_id2", "node_address2"] ]

Search available models
-----------------------

| **URL** : ``/search-available-models``
| **Description** : Get available models on the grid network.
| **Method** : ``GET``
| **Auth required** : NO (can be changed)

Status Code: 200 OK
'''''''''''''''''''

.. code:: json

    [ "model_id1", "model_id2", "model_id3" ]

Search available tags
---------------------

| **URL** : ``/search-available-tags``
| **Description** : Get all available tensor tags on the grid network.
| **Method** : ``GET``
| **Auth required** : NO (can be changed)

Status Code: 200 OK
'''''''''''''''''''

.. code:: json

    [ "#tensor_tag1", "#tensor_tag2", "#tensor_tag3" ]

Search Tags
-----------

| **URL** : ``/search``
| **Description** : Search specific tags on grid network.
| **Method** : ``POST``
| **Content-Type** : application/json **Auth required** : NO (can be
changed)

Request Body
^^^^^^^^^^^^

.. code:: json

    {
        "query" : ["#tensor_tag1", "#tensor-tag2", "#tensor-tag3"]
    }

Status Code: 200 OK
'''''''''''''''''''

.. code:: json

    [ ["node_id1","node_address1"], ["node_id2", "node_address2"] ]

Status Code: 400 Bad Request
''''''''''''''''''''''''''''

.. code:: json

    {
        "message": "Invalid json fields."
    }

