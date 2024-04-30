What is this?
-------------
This document lists down the features missing in mongomock library. PRs for these features are highly appreciated.

If I miss to include a feature in the below list, Please feel free to add to the below list and raise a PR.

* $rename complex operations - https://docs.mongodb.com/manual/reference/operator/update/rename/
* create_collection options - https://docs.mongodb.com/v3.2/reference/method/db.createCollection/#definition
* bypass_document_validation options
* session options
* codec options
* Operations of the aggregate pipeline:
  * `$bucketAuto <https://docs.mongodb.com/manual/reference/operator/aggregation/bucketAuto/>`_
  * `$collStats <https://docs.mongodb.com/manual/reference/operator/aggregation/collStats/>`_
  * `$currentOp <https://docs.mongodb.com/manual/reference/operator/aggregation/currentOp/>`_
  * `$geoNear <https://docs.mongodb.com/manual/reference/operator/aggregation/geoNear/>`_
  * `$indexStats <https://docs.mongodb.com/manual/reference/operator/aggregation/indexStats/>`_
  * `$listLocalSessions <https://docs.mongodb.com/manual/reference/operator/aggregation/listLocalSessions/>`_
  * `$listSessions <https://docs.mongodb.com/manual/reference/operator/aggregation/listSessions/>`_
  * `$merge <https://docs.mongodb.com/manual/reference/operator/aggregation/merge/>`_
  * `$planCacheStats <https://docs.mongodb.com/manual/reference/operator/aggregation/planCacheStats/>`_
  * `$redact <https://docs.mongodb.com/manual/reference/operator/aggregation/redact/>`_
  * `$replaceWith <https://docs.mongodb.com/manual/reference/operator/aggregation/replaceWith/>`_
  * `$sortByCount <https://docs.mongodb.com/manual/reference/operator/aggregation/sortByCount/>`_
  * `$unset <https://docs.mongodb.com/manual/reference/operator/aggregation/unset/>` _
* Operators within the aggregate pipeline:
  * Arithmetic operations on dates:
    * `$add <https://docs.mongodb.com/manual/reference/operator/aggregation/add/>`_
  * Some date operators ($isoDayOfWeek, $isoWeekYear, …)
  * Some set operators ($setIntersection, $setDifference, …)
  * Some string operators ($indexOfBytes, $trim, …)
  * Text search operator ($meta)
  * Projection operator $map
  * Array operators ($isArray, $indexOfArray, …)
  * `$mergeObjects <https://docs.mongodb.com/manual/reference/operator/aggregation/mergeObjects/>`_
  * Some type conversion operators ($convert, …)
* Operators within the query language (find):
  * `$jsonSchema <https://docs.mongodb.com/manual/reference/operator/query/jsonSchema/>`_
  * `$text <https://docs.mongodb.com/manual/reference/operator/query/text/>`_ search
  * `$where <https://docs.mongodb.com/manual/reference/operator/query/where/>`_
* `map_reduce <https://docs.mongodb.com/manual/reference/command/mapReduce/>`_ options (``scope`` and ``finalize``)
* Database `command <https://docs.mongodb.com/manual/reference/command/>`_ method except for the ``ping`` command.
* Raw Batch BSON operations (`aggregate_raw_batches` and `find_raw_batches`)
* `Expiring Data <https://docs.mongodb.com/manual/tutorial/expire-data/>`_
