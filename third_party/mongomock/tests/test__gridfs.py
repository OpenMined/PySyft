# stdlib
import os
import time
import unittest
from unittest import TestCase
from unittest import skipIf
from unittest import skipUnless

# third party
import mongomock
from mongomock import helpers
import mongomock.gridfs
from packaging import version

try:
    # third party
    import gridfs
    from gridfs import errors

    _HAVE_GRIDFS = True
except ImportError:
    _HAVE_GRIDFS = False


try:
    # third party
    from bson.objectid import ObjectId
    import pymongo
    from pymongo import MongoClient as PymongoClient
except ImportError:
    ...


@skipUnless(helpers.HAVE_PYMONGO, "pymongo not installed")
@skipUnless(
    _HAVE_GRIDFS and hasattr(gridfs.__builtins__, "copy"), "gridfs not installed"
)
@skipIf(os.getenv("NO_LOCAL_MONGO"), "No local Mongo server running")
class GridFsTest(TestCase):
    @classmethod
    def setUpClass(cls):
        mongomock.gridfs.enable_gridfs_integration()

    def setUp(self):
        super(GridFsTest, self).setUp()
        self.fake_conn = mongomock.MongoClient()
        self.mongo_conn = self._connect_to_local_mongodb()
        self.db_name = "mongomock___testing_db"

        self.mongo_conn[self.db_name]["fs"]["files"].drop()
        self.mongo_conn[self.db_name]["fs"]["chunks"].drop()

        self.real_gridfs = gridfs.GridFS(self.mongo_conn[self.db_name])
        self.fake_gridfs = gridfs.GridFS(self.fake_conn[self.db_name])

    def tearDown(self):
        super(GridFsTest, self).setUp()
        self.mongo_conn.close()
        self.fake_conn.close()

    def test__put_get_small(self):
        before = time.time()
        fid = self.fake_gridfs.put(GenFile(50))
        rid = self.real_gridfs.put(GenFile(50))
        after = time.time()
        ffile = self.fake_gridfs.get(fid)
        rfile = self.real_gridfs.get(rid)
        self.assertEqual(ffile.read(), rfile.read())
        fake_doc = self.get_fake_file(fid)
        mongo_doc = self.get_mongo_file(rid)
        self.assertSameFile(mongo_doc, fake_doc, max_delta_seconds=after - before + 1)

    def test__put_get_big(self):
        # 500k files are bigger than doc size limit
        before = time.time()
        fid = self.fake_gridfs.put(GenFile(500000, 10))
        rid = self.real_gridfs.put(GenFile(500000, 10))
        after = time.time()
        ffile = self.fake_gridfs.get(fid)
        rfile = self.real_gridfs.get(rid)
        self.assertEqual(ffile.read(), rfile.read())
        fake_doc = self.get_fake_file(fid)
        mongo_doc = self.get_mongo_file(rid)
        self.assertSameFile(mongo_doc, fake_doc, max_delta_seconds=after - before + 1)

    def test__delete_exists_small(self):
        fid = self.fake_gridfs.put(GenFile(50))
        self.assertTrue(self.get_fake_file(fid) is not None)
        self.assertTrue(self.fake_gridfs.exists(fid))
        self.fake_gridfs.delete(fid)
        self.assertFalse(self.fake_gridfs.exists(fid))
        self.assertFalse(self.get_fake_file(fid) is not None)
        # All the chunks got removed
        self.assertEqual(0, self.fake_conn[self.db_name].fs.chunks.count_documents({}))

    def test__delete_exists_big(self):
        fid = self.fake_gridfs.put(GenFile(500000))
        self.assertTrue(self.get_fake_file(fid) is not None)
        self.assertTrue(self.fake_gridfs.exists(fid))
        self.fake_gridfs.delete(fid)
        self.assertFalse(self.fake_gridfs.exists(fid))
        self.assertFalse(self.get_fake_file(fid) is not None)
        # All the chunks got removed
        self.assertEqual(0, self.fake_conn[self.db_name].fs.chunks.count_documents({}))

    def test__delete_no_file(self):
        # Just making sure we don't crash
        self.fake_gridfs.delete(ObjectId())

    def test__list_files(self):
        fids = [
            self.fake_gridfs.put(GenFile(50, 9), filename="one"),
            self.fake_gridfs.put(GenFile(62, 5), filename="two"),
            self.fake_gridfs.put(GenFile(654, 1), filename="three"),
            self.fake_gridfs.put(GenFile(5), filename="four"),
        ]
        names = ["one", "two", "three", "four"]
        names_no_two = [x for x in names if x != "two"]
        for x in self.fake_gridfs.list():
            self.assertIn(x, names)

        self.fake_gridfs.delete(fids[1])

        for x in self.fake_gridfs.list():
            self.assertIn(x, names_no_two)

        three_file = self.get_fake_file(fids[2])
        self.assertEqual("three", three_file["filename"])
        self.assertEqual(654, three_file["length"])
        self.fake_gridfs.delete(fids[0])
        self.fake_gridfs.delete(fids[2])
        self.fake_gridfs.delete(fids[3])
        self.assertEqual(0, len(self.fake_gridfs.list()))

    def test__find_files(self):
        fids = [
            self.fake_gridfs.put(GenFile(50, 9), filename="a"),
            self.fake_gridfs.put(GenFile(62, 5), filename="b"),
            self.fake_gridfs.put(GenFile(654, 1), filename="b"),
            self.fake_gridfs.put(GenFile(5), filename="a"),
        ]
        c = self.fake_gridfs.find({"filename": "a"}).sort("uploadDate", -1)
        should_be_fid3 = c.next()
        should_be_fid0 = c.next()
        self.assertFalse(c.alive)

        self.assertEqual(fids[3], should_be_fid3._id)
        self.assertEqual(fids[0], should_be_fid0._id)

    def test__put_exists(self):
        self.fake_gridfs.put(GenFile(1), _id="12345")
        with self.assertRaises(errors.FileExists):
            self.fake_gridfs.put(GenFile(2, 3), _id="12345")

    def assertSameFile(self, real, fake, max_delta_seconds=1):
        # https://pymongo.readthedocs.io/en/stable/migrate-to-pymongo4.html#disable-md5-parameter-is-removed
        if helpers.PYMONGO_VERSION < version.parse("4.0"):
            self.assertEqual(real["md5"], fake["md5"])

        self.assertEqual(real["length"], fake["length"])
        self.assertEqual(real["chunkSize"], fake["chunkSize"])
        self.assertLessEqual(
            abs(real["uploadDate"] - fake["uploadDate"]).seconds,
            max_delta_seconds,
            msg="real: %s, fake: %s" % (real["uploadDate"], fake["uploadDate"]),
        )

    def get_mongo_file(self, i):
        return self.mongo_conn[self.db_name]["fs"]["files"].find_one({"_id": i})

    def get_fake_file(self, i):
        return self.fake_conn[self.db_name]["fs"]["files"].find_one({"_id": i})

    def _connect_to_local_mongodb(self, num_retries=60):
        """Performs retries on connection refused errors (for travis-ci builds)"""
        for retry in range(num_retries):
            if retry > 0:
                time.sleep(0.5)
            try:
                return PymongoClient(
                    host=os.environ.get("TEST_MONGO_HOST", "localhost"), maxPoolSize=1
                )
            except pymongo.errors.ConnectionFailure as e:
                if retry == num_retries - 1:
                    raise
                if "connection refused" not in e.message.lower():
                    raise


class GenFile(object):
    def __init__(self, length, value=0, do_encode=True):
        self.gen = self._gen_data(length, value)
        self.do_encode = do_encode

    def _gen_data(self, length, value):
        while length:
            length -= 1
            yield value

    def _maybe_encode(self, s):
        if self.do_encode and isinstance(s, str):
            return s.encode("UTF-8")
        return s

    def read(self, num_bytes=-1):
        s = ""
        if num_bytes <= 0:
            bytes_left = -1
        else:
            bytes_left = num_bytes
        while True:
            n = next(self.gen, None)
            if n is None:
                return self._maybe_encode(s)
            s += chr(n)
            bytes_left -= 1
            if bytes_left == 0:
                return self._maybe_encode(s)


if __name__ == "__main__":
    unittest.main()
