# stdlib
import unittest

# third party
import mongomock


class NotImplementedTests(unittest.TestCase):
    def tearDown(self):
        mongomock.warn_on_feature("session")

    def test_raises(self):
        collection = mongomock.MongoClient().db.collection
        with self.assertRaises(NotImplementedError):
            collection.insert_one({}, session=True)

    def test_ignores(self):
        mongomock.ignore_feature("session")

        collection = mongomock.MongoClient().db.collection
        collection.insert_one({}, session=True)

    def test_on_and_off(self):
        collection = mongomock.MongoClient().db.collection

        with self.assertRaises(NotImplementedError):
            collection.insert_one({"_id": 1}, session=True)

        mongomock.ignore_feature("session")

        collection.insert_one({"_id": 2}, session=True)

        mongomock.warn_on_feature("session")

        with self.assertRaises(NotImplementedError):
            collection.insert_one({"_id": 3}, session=True)

        self.assertEqual({2}, {doc["_id"] for doc in collection.find()})

    def test_wrong_key(self):
        with self.assertRaises(KeyError):
            mongomock.ignore_feature("sessions")
