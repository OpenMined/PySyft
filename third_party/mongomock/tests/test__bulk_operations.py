# stdlib
import os

# third party
import mongomock
from mongomock import helpers
from packaging import version

try:
    # stdlib
    from unittest import mock

    _HAVE_MOCK = True
except ImportError:
    try:
        # third party
        import mock

        _HAVE_MOCK = True
    except ImportError:
        _HAVE_MOCK = False

try:
    # third party
    import pymongo
except ImportError:
    pymongo = None

# stdlib
from unittest import TestCase
from unittest import skipIf

# third party
from tests.multicollection import MultiCollection


# https://pymongo.readthedocs.io/en/stable/migrate-to-pymongo4.html#collection-initialize-ordered-bulk-op-and-initialize-unordered-bulk-op-is-removed
@skipIf(helpers.PYMONGO_VERSION >= version.parse("4.0"), "pymongo v4 or above")
class BulkOperationsTest(TestCase):
    test_with_pymongo = False

    def setUp(self):
        super(BulkOperationsTest, self).setUp()
        if self.test_with_pymongo:
            self.client = pymongo.MongoClient(
                host=os.environ.get("TEST_MONGO_HOST", "localhost")
            )
        else:
            self.client = mongomock.MongoClient()
        self.db = self.client["somedb"]
        self.db.collection.drop()
        for _i in "abx":
            self.db.collection.create_index(
                _i, unique=False, name="idx" + _i, sparse=True, background=True
            )
        self.bulk_op = self.db.collection.initialize_ordered_bulk_op()

    def __check_document(self, doc, count=1):
        found_num = self.db.collection.find(doc).count()
        if found_num != count:
            all = list(self.db.collection.find())
            self.fail(
                "Document %s count()=%s BUT expected count=%s! All"
                " documents: %s" % (doc, found_num, count, all)
            )

    def __check_result(self, result, **expecting_values):
        for key in (
            "nModified",
            "nUpserted",
            "nMatched",
            "writeErrors",
            "upserted",
            "writeConcernErrors",
            "nRemoved",
            "nInserted",
        ):
            exp_val = expecting_values.get(key)
            has_val = result.get(key)
            if self.test_with_pymongo and key == "nModified" and has_val is None:
                # ops, real pymongo did not returned 'nModified' key!
                continue
            self.assertFalse(
                has_val is None, "Missed key '%s' in result: %s" % (key, result)
            )
            if exp_val:
                self.assertEqual(
                    exp_val,
                    has_val,
                    "Invalid result %s=%s (but expected value=%s)"
                    % (key, has_val, exp_val),
                )
            else:
                self.assertFalse(
                    bool(has_val), "Received unexpected value %s = %s" % (key, has_val)
                )

    def __execute_and_check_result(self, write_concern=None, **expecting_result):
        result = self.bulk_op.execute(write_concern=write_concern)
        self.__check_result(result, **expecting_result)

    def __check_number_of_elements(self, count):
        has_count = self.db.collection.count()
        self.assertEqual(
            has_count,
            count,
            "There is %s documents but there should be %s" % (has_count, count),
        )

    def test__insert(self):
        self.bulk_op.insert({"a": 1, "b": 2})
        self.bulk_op.insert({"a": 2, "b": 4})
        self.bulk_op.insert({"a": 2, "b": 6})

        self.__check_number_of_elements(0)
        self.__execute_and_check_result(nInserted=3)
        self.__check_document({"a": 1, "b": 2})
        self.__check_document({"a": 2, "b": 4})
        self.__check_document({"a": 2, "b": 6})

    def test__bulk_update_must_raise_error_if_missed_operator(self):
        self.assertRaises(ValueError, self.bulk_op.find({"a": 1}).update, {"b": 20})

    def test__bulk_execute_must_raise_error_if_bulk_empty(self):
        self.assertRaises(mongomock.InvalidOperation, self.bulk_op.execute)

    def test_update(self):
        self.bulk_op.find({"a": 1}).update({"$set": {"b": 20}})
        self.__execute_and_check_result()
        self.__check_number_of_elements(0)

    def test__update_must_update_all_documents(self):
        self.db.collection.insert_one({"a": 1, "b": 2})
        self.db.collection.insert_one({"a": 2, "b": 4})
        self.db.collection.insert_one({"a": 2, "b": 8})

        self.bulk_op.find({"a": 1}).update({"$set": {"b": 20}})
        self.bulk_op.find({"a": 2}).update({"$set": {"b": 40}})

        self.__check_document({"a": 1, "b": 2})
        self.__check_document({"a": 2, "b": 4})
        self.__check_document({"a": 2, "b": 8})

        self.__execute_and_check_result(nMatched=3, nModified=3)
        self.__check_document({"a": 1, "b": 20})
        self.__check_document({"a": 2, "b": 40}, 2)

    def test__ordered_insert_and_update(self):
        self.bulk_op.insert({"a": 1, "b": 2})
        self.bulk_op.find({"a": 1}).update({"$set": {"b": 3}})
        self.__execute_and_check_result(nInserted=1, nMatched=1, nModified=1)
        self.__check_document({"a": 1, "b": 3})

    def test__update_one(self):
        self.db.collection.insert_one({"a": 2, "b": 1})
        self.db.collection.insert_one({"a": 2, "b": 2})

        self.bulk_op.find({"a": 2}).update_one({"$set": {"b": 3}})
        self.__execute_and_check_result(nMatched=1, nModified=1)
        self.__check_document({"a": 2}, count=2)
        self.__check_number_of_elements(2)

    def test__remove(self):
        self.db.collection.insert_one({"a": 2, "b": 1})
        self.db.collection.insert_one({"a": 2, "b": 2})

        self.bulk_op.find({"a": 2}).remove()

        self.__execute_and_check_result(nRemoved=2)
        self.__check_number_of_elements(0)

    def test__remove_one(self):
        self.db.collection.insert_one({"a": 2, "b": 1})
        self.db.collection.insert_one({"a": 2, "b": 2})

        self.bulk_op.find({"a": 2}).remove_one()

        self.__execute_and_check_result(nRemoved=1)
        self.__check_document({"a": 2}, 1)
        self.__check_number_of_elements(1)

    @skipIf(not _HAVE_MOCK, "The mock library is not installed")
    def test_upsert_replace_one_on_empty_set(self):
        self.bulk_op.find({}).upsert().replace_one({"x": 1})
        self.__execute_and_check_result(
            nUpserted=1, upserted=[{"index": 0, "_id": mock.ANY}]
        )

    def test_upsert_replace_one(self):
        self.db.collection.insert_one({"a": 2, "b": 1})
        self.db.collection.insert_one({"a": 2, "b": 2})
        self.bulk_op.find({"a": 2}).replace_one({"x": 1})
        self.__execute_and_check_result(nModified=1, nMatched=1)
        self.__check_document({"a": 2}, 1)
        self.__check_document({"x": 1}, 1)
        self.__check_number_of_elements(2)

    @skipIf(not _HAVE_MOCK, "The mock library is not installed")
    def test_upsert_update_on_empty_set(self):
        self.bulk_op.find({}).upsert().update({"$set": {"a": 1, "b": 2}})
        self.__execute_and_check_result(
            nUpserted=1, upserted=[{"index": 0, "_id": mock.ANY}]
        )
        self.__check_document({"a": 1, "b": 2})
        self.__check_number_of_elements(1)

    def test_upsert_update(self):
        self.db.collection.insert_one({"a": 2, "b": 1})
        self.db.collection.insert_one({"a": 2, "b": 2})
        self.bulk_op.find({"a": 2}).upsert().update({"$set": {"b": 3}})
        self.__execute_and_check_result(nMatched=2, nModified=2)
        self.__check_document({"a": 2, "b": 3}, 2)
        self.__check_number_of_elements(2)

    def test_upsert_update_one(self):
        self.db.collection.insert_one({"a": 2, "b": 1})
        self.db.collection.insert_one({"a": 2, "b": 1})
        self.bulk_op.find({"a": 2}).upsert().update_one({"$inc": {"b": 1, "x": 1}})
        self.__execute_and_check_result(nModified=1, nMatched=1)
        self.__check_document({"a": 2, "b": 1}, 1)
        self.__check_document({"a": 2, "b": 2, "x": 1}, 1)
        self.__check_number_of_elements(2)


@skipIf(not helpers.HAVE_PYMONGO, "pymongo not installed")
@skipIf(os.getenv("NO_LOCAL_MONGO"), "No local Mongo server running")
class BulkOperationsWithPymongoTest(BulkOperationsTest):
    test_with_pymongo = True


@skipIf(not helpers.HAVE_PYMONGO, "pymongo not installed")
@skipIf(os.getenv("NO_LOCAL_MONGO"), "No local Mongo server running")
# https://pymongo.readthedocs.io/en/stable/migrate-to-pymongo4.html#collection-initialize-ordered-bulk-op-and-initialize-unordered-bulk-op-is-removed
@skipIf(helpers.PYMONGO_VERSION >= version.parse("4.0"), "pymongo v4 or above")
class CollectionComparisonTest(TestCase):
    def setUp(self):
        super(CollectionComparisonTest, self).setUp()
        self.fake_conn = mongomock.MongoClient()
        self.mongo_conn = pymongo.MongoClient(
            host=os.environ.get("TEST_MONGO_HOST", "localhost")
        )
        self.db_name = "mongomock___testing_db"
        self.collection_name = "mongomock___testing_collection"
        self.mongo_conn[self.db_name][self.collection_name].remove()
        self.cmp = MultiCollection(
            {
                "fake": self.fake_conn[self.db_name][self.collection_name],
                "real": self.mongo_conn[self.db_name][self.collection_name],
            }
        )
        self.bulks = MultiCollection(
            {
                "fake": self.cmp.conns["fake"].initialize_ordered_bulk_op(),
                "real": self.cmp.conns["real"].initialize_ordered_bulk_op(),
            }
        )

        # hacky! Depending on mongo server version 'nModified' is returned or not..
        # so let make simple bulk operation to know what's the server behaviour...
        coll = self.mongo_conn[self.db_name]["mongomock_testing_prepare_test"]
        bulk = coll.initialize_ordered_bulk_op()
        bulk.insert({"a": 1})
        insert_returns_nmodified = "nModified" in bulk.execute()

        bulk = self.cmp.conns["real"].initialize_ordered_bulk_op()
        bulk.find({"a": 1}).update({"$set": {"a": 2}})
        update_returns_nmodified = "nModified" in bulk.execute()
        coll.drop()

        self.bulks.conns["fake"]._set_nModified_policy(
            insert_returns_nmodified, update_returns_nmodified
        )

    def test__insert(self):
        self.bulks.do.insert({"a": 1, "b": 1})
        self.bulks.do.insert({"a": 2, "b": 2})
        self.bulks.do.insert({"a": 2, "b": 2})
        self.bulks.compare.execute()

    def test__mixed_operations(self):
        self.cmp.do.insert({"a": 1, "b": 3})
        self.cmp.do.insert({"a": 2, "c": 1})
        self.cmp.do.insert({"a": 2, "c": 2})
        self.cmp.do.insert({"a": 3, "c": 1})
        self.cmp.do.insert({"a": 4, "d": 2})
        self.cmp.do.insert({"a": 5, "d": 11})
        self.cmp.do.insert({"a": 5, "d": 22})

        self.bulks.do.insert({"a": 1, "b": 1})
        for bwo in self.bulks.do.find({"a": 2}).values():
            bwo.remove_one()
        for bwo in self.bulks.do.find({"a": 3}).values():
            bwo.update({"$inc": {"b": 1}})
        for bwo in self.bulks.do.find({"a": 4}).values():
            bwo.upsert().replace_one({"b": 11, "x": "y"})
        for bwo in self.bulks.do.find({"a": 5}).values():
            bwo.upsert().update({"$inc": {"b": 11}})
        self.bulks.compare.execute()
        self.cmp.compare.find(sort=[("a", 1), ("b", 1), ("c", 1), ("d", 1)])
