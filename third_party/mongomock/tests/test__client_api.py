# stdlib
import sys
import unittest
from unittest import skipIf
from unittest import skipUnless

# third party
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

# third party
import mongomock
from mongomock import helpers

try:
    # third party
    from bson import codec_options
    from pymongo.read_preferences import ReadPreference
except ImportError:
    pass


class MongoClientApiTest(unittest.TestCase):
    def test__read_preference(self):
        client = mongomock.MongoClient()
        self.assertEqual("Primary", client.read_preference.name)
        self.assertEqual(client.read_preference, client.db.read_preference)
        self.assertEqual(client.read_preference, client.db.coll.read_preference)

        client2 = mongomock.MongoClient(read_preference=client.read_preference)
        self.assertEqual(client2.read_preference, client.read_preference)

        with self.assertRaises(TypeError):
            mongomock.MongoClient(read_preference=0)

    @unittest.skipIf(not helpers.HAVE_PYMONGO, "pymongo not installed")
    def test__different_read_preference(self):
        client = mongomock.MongoClient(read_preference=ReadPreference.NEAREST)
        self.assertEqual(ReadPreference.NEAREST, client.db.read_preference)
        self.assertEqual(ReadPreference.NEAREST, client.db.coll.read_preference)

    @unittest.skipIf(not helpers.HAVE_PYMONGO, "pymongo not installed")
    def test__codec_options_with_pymongo(self):
        client = mongomock.MongoClient()
        self.assertEqual(codec_options.CodecOptions(), client.codec_options)
        self.assertFalse(client.codec_options.tz_aware)

    def test__codec_options(self):
        client = mongomock.MongoClient()
        self.assertFalse(client.codec_options.tz_aware)

        client = mongomock.MongoClient(tz_aware=True)
        self.assertTrue(client.codec_options.tz_aware)
        self.assertTrue(client.db.collection.codec_options.tz_aware)

        with self.assertRaises(TypeError):
            mongomock.MongoClient(tz_aware="True")

    def test__parse_url(self):
        client = mongomock.MongoClient("mongodb://localhost:27017/")
        self.assertEqual(("localhost", 27017), client.address)

        client = mongomock.MongoClient("mongodb://localhost:1234,example.com/")
        self.assertEqual(("localhost", 1234), client.address)

        client = mongomock.MongoClient("mongodb://example.com,localhost:1234/")
        self.assertEqual(("example.com", 27017), client.address)

        client = mongomock.MongoClient("mongodb://[::1]:1234/")
        self.assertEqual(("::1", 1234), client.address)

        with self.assertRaises(ValueError):
            mongomock.MongoClient("mongodb://localhost:1234:456/")

        with self.assertRaises(ValueError):
            mongomock.MongoClient("mongodb://localhost:123456/")

        with self.assertRaises(ValueError):
            mongomock.MongoClient("mongodb://localhost:mongoport/")

    def test__equality(self):
        self.assertEqual(
            mongomock.MongoClient("mongodb://localhost:27017/"),
            mongomock.MongoClient("mongodb://localhost:27017/"),
        )
        self.assertEqual(
            mongomock.MongoClient("mongodb://localhost:27017/"),
            mongomock.MongoClient("localhost"),
        )
        self.assertNotEqual(
            mongomock.MongoClient("/var/socket/mongo.sock"),
            mongomock.MongoClient("localhost"),
        )

    @skipIf(
        sys.version_info < (3,),
        "Older versions of Python do not handle hashing the same way",
    )
    @skipUnless(
        helpers.PYMONGO_VERSION < version.parse("3.12"),
        "older versions of pymongo didn't have proper hashing",
    )
    def test__not_hashable(self):
        with self.assertRaises(TypeError):
            {mongomock.MongoClient("localhost")}  # pylint: disable=expression-not-assigned

    @skipIf(
        sys.version_info < (3,),
        "Older versions of Python do not handle hashing the same way",
    )
    @skipIf(
        helpers.PYMONGO_VERSION < version.parse("3.12"),
        "older versions of pymongo didn't have proper hashing",
    )
    def test__hashable(self):
        {mongomock.MongoClient("localhost")}  # pylint: disable=expression-not-assigned

    def test__parse_hosts(self):
        client = mongomock.MongoClient("localhost")
        self.assertEqual(("localhost", 27017), client.address)

        client = mongomock.MongoClient("localhost:1234,example.com")
        self.assertEqual(("localhost", 1234), client.address)

        client = mongomock.MongoClient("example.com,localhost:1234")
        self.assertEqual(("example.com", 27017), client.address)

        client = mongomock.MongoClient("[::1]:1234")
        self.assertEqual(("::1", 1234), client.address)

        client = mongomock.MongoClient("/var/socket/mongo.sock")
        self.assertEqual(("/var/socket/mongo.sock", None), client.address)

        with self.assertRaises(ValueError):
            mongomock.MongoClient("localhost:1234:456")

        with self.assertRaises(ValueError):
            mongomock.MongoClient("localhost:123456")

        with self.assertRaises(ValueError):
            mongomock.MongoClient("localhost:mongoport")

    @unittest.skipIf(not _HAVE_MOCK, "mock not installed")
    def test_database_names(self):
        client = mongomock.MongoClient()
        client.one_db.my_collec.insert_one({})

        if helpers.PYMONGO_VERSION >= version.parse("4.0"):
            with self.assertRaises(TypeError):
                client.database_names()
            return

        with mock.patch("warnings.warn") as mock_warn:
            self.assertEqual(["one_db"], client.database_names())

        self.assertEqual(1, mock_warn.call_count)
        self.assertIn("deprecated", mock_warn.call_args[0][0])

    def test_list_database_names(self):
        client = mongomock.MongoClient()
        self.assertEqual([], client.list_database_names())

        # Query a non existant collection.
        client.one_db.my_collec.find_one()
        self.assertEqual([], client.list_database_names())

        client.one_db.my_collec.insert_one({})
        self.assertEqual(["one_db"], client.list_database_names())

    def test_client_implements_context_managers(self):
        with mongomock.MongoClient() as client:
            client.one_db.my_collec.insert_one({})
            result = client.one_db.my_collec.find_one({})
            self.assertTrue(result)

    def test_start_session(self):
        client = mongomock.MongoClient()
        with self.assertRaises(NotImplementedError):
            client.start_session()

    @mock.patch("mongomock.SERVER_VERSION", "3.6")
    def test_server_version(self):
        client = mongomock.MongoClient()
        server_info = client.server_info()
        self.assertEqual("3.6", server_info["version"])
        self.assertEqual([3, 6, 0, 0], server_info["versionArray"])

    def test_consistent_server_version(self):
        client = mongomock.MongoClient()
        server_info = client.server_info()
        with mock.patch("mongomock.SERVER_VERSION", "3.6"):
            self.assertEqual(server_info, client.server_info())
