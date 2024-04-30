# stdlib
import collections
import datetime
import sys
from unittest import TestCase
from unittest import skipIf
from unittest import skipUnless
from uuid import uuid4

# third party
import mongomock
from mongomock import helpers
from mongomock import read_concern
from packaging import version

try:
    # third party
    from bson import codec_options
    from pymongo.read_preferences import ReadPreference
except ImportError:
    pass


class UTCPlus2(datetime.tzinfo):
    def fromutc(self, dt):
        return dt + self.utcoffset(dt)

    def tzname(self, dt):
        return "<dummy UTC+2>"

    def utcoffset(self, dt):
        return datetime.timedelta(hours=2)

    def dst(self, dt):
        return datetime.timedelta()


class DatabaseAPITest(TestCase):
    def setUp(self):
        self.database = mongomock.MongoClient().somedb

    def test__get_collection_by_attribute_underscore(self):
        with self.assertRaises(AttributeError) as err_context:
            self.database._users  # pylint: disable=pointless-statement

        self.assertIn("Database has no attribute '_users'", str(err_context.exception))

        # No problem accessing it through __get_item__.
        self.database["_users"].insert_one({"a": 1})
        self.assertEqual(1, self.database["_users"].find_one().get("a"))

    def test__session(self):
        with self.assertRaises(NotImplementedError):
            self.database.list_collection_names(session=1)
        with self.assertRaises(NotImplementedError):
            self.database.drop_collection("a", session=1)
        with self.assertRaises(NotImplementedError):
            self.database.create_collection("a", session=1)
        with self.assertRaises(NotImplementedError):
            self.database.dereference(_DBRef("somedb", "a", "b"), session=1)

    def test__command_ping(self):
        self.assertEqual({"ok": 1}, self.database.command({"ping": 1}))

    def test__command_ping_string(self):
        self.assertEqual({"ok": 1}, self.database.command("ping"))

    def test__command_fake_ping_string(self):
        with self.assertRaises(NotImplementedError):
            self.assertEqual({"ok": 1}, self.database.command("a_nice_ping"))

    def test__command(self):
        with self.assertRaises(NotImplementedError):
            self.database.command({"count": "user"})

    def test__repr(self):
        self.assertEqual(
            "Database(mongomock.MongoClient('localhost', 27017), 'somedb')",
            repr(self.database),
        )

    def test__rename_unknown_collection(self):
        with self.assertRaises(mongomock.OperationFailure):
            self.database.rename_collection("a", "b")

    def test__dereference(self):
        self.database.a.insert_one({"_id": "b", "val": 42})
        doc = self.database.dereference(_DBRef("somedb", "a", "b"))
        self.assertEqual({"_id": "b", "val": 42}, doc)

        self.assertEqual(None, self.database.dereference(_DBRef("somedb", "a", "a")))
        self.assertEqual(None, self.database.dereference(_DBRef("somedb", "b", "b")))

        with self.assertRaises(ValueError):
            self.database.dereference(_DBRef("otherdb", "a", "b"))

        with self.assertRaises(TypeError):
            self.database.dereference("b")

    def test__read_preference(self):
        self.assertEqual("Primary", self.database.read_preference.name)
        self.assertEqual(
            self.database.collection.read_preference, self.database.read_preference
        )

        with self.assertRaises(TypeError):
            self.database.get_collection("a", read_preference="nearest")

    @skipIf(not helpers.HAVE_PYMONGO, "pymongo not installed")
    def test__get_collection_different_read_preference(self):
        database = mongomock.MongoClient().get_database(
            "somedb", read_preference=ReadPreference.NEAREST
        )
        self.assertEqual("Nearest", database.read_preference.name)
        self.assertEqual(database.read_preference, database.collection.read_preference)

        col = database.get_collection("col", read_preference=ReadPreference.PRIMARY)
        self.assertEqual("Primary", col.read_preference.name)

    @skipIf(not helpers.HAVE_PYMONGO, "pymongo not installed")
    def test__get_collection_different_codec_options(self):
        database = mongomock.MongoClient().somedb
        a = database.get_collection(
            "a", codec_options=codec_options.CodecOptions(tz_aware=True)
        )
        self.assertTrue(a.codec_options.tz_aware)

    @skipIf(not helpers.HAVE_PYMONGO, "pymongo not installed")
    def test__codec_options(self):
        self.assertEqual(codec_options.CodecOptions(), self.database.codec_options)

    @skipIf(not helpers.HAVE_PYMONGO, "pymongo not installed")
    def test__read_concern(self):
        self.assertEqual(read_concern.ReadConcern(), self.database.read_concern)

    def test__with_options(self):
        with self.assertRaises(NotImplementedError):
            self.database.with_options(write_concern=3)

    @skipIf(not helpers.HAVE_PYMONGO, "pymongo not installed")
    def test__with_options_pymongo(self):
        other = self.database.with_options(read_preference=self.database.NEAREST)
        self.assertFalse(other is self.database)

        self.database.coll.insert_one({"_id": 42})
        self.assertEqual({"_id": 42}, other.coll.find_one())

        self.database.with_options(codec_options=codec_options.CodecOptions())
        self.database.with_options()

        self.database.with_options(
            codec_options=codec_options.CodecOptions(tz_aware=True)
        )

        tz_aware_db = mongomock.MongoClient(tz_aware=True).somedb
        self.assertIs(
            tz_aware_db,
            tz_aware_db.with_options(
                codec_options=codec_options.CodecOptions(tz_aware=True)
            ),
        )

        custom_document_class = codec_options.CodecOptions(
            document_class=collections.OrderedDict
        )
        with self.assertRaises(NotImplementedError):
            self.database.with_options(custom_document_class)

        custom_uuid_representation = codec_options.CodecOptions(uuid_representation=4)
        db = self.database
        col = db.get_collection(
            "yes_hello", codec_options=custom_uuid_representation
        ).insert_one({"_id": uuid4()})

        custom_unicode_error_hander = codec_options.CodecOptions(
            unicode_decode_error_handler="ignore"
        )
        with self.assertRaises(NotImplementedError):
            self.database.with_options(custom_unicode_error_hander)

        custom_tzinfo = codec_options.CodecOptions(tz_aware=True, tzinfo=UTCPlus2())
        with self.assertRaises(NotImplementedError):
            self.database.with_options(custom_tzinfo)

    def test__collection_names(self):
        self.database.create_collection("a")
        self.database.create_collection("b")

        if helpers.PYMONGO_VERSION >= version.parse("4.0"):
            with self.assertRaises(TypeError):
                self.database.collection_names()
            return

        self.assertEqual(set(self.database.collection_names()), set(["a", "b"]))

        self.database.c.drop()
        self.assertEqual(set(self.database.collection_names()), set(["a", "b"]))

    def test__list_collection_names(self):
        self.database.create_collection("a")
        self.database.create_collection("b")
        self.assertEqual(set(self.database.list_collection_names()), set(["a", "b"]))

        self.database.c.drop()
        self.assertEqual(set(self.database.list_collection_names()), set(["a", "b"]))

    def test__list_collections(self):
        self.database.create_collection("a")

        with self.assertRaises(NotImplementedError):
            self.database.list_collections()

    def test__create_collection(self):
        coll = self.database.create_collection("c")
        self.assertIs(self.database.c, coll)
        self.assertRaises(
            mongomock.CollectionInvalid, self.database.create_collection, "c"
        )

    def test__create_collection_bad_names(self):
        with self.assertRaises(TypeError):
            self.database.create_collection(3)
        with self.assertRaises(TypeError):
            self.database[3]  # pylint: disable=pointless-statement

        bad_names = (
            "",
            "foo..bar",
            "...",
            "$foo",
            ".foo",
            "bar.",
            "foo\x00bar",
        )
        for name in bad_names:
            with self.assertRaises(mongomock.InvalidName, msg=name):
                self.database.create_collection(name)
            with self.assertRaises(mongomock.InvalidName, msg=name):
                self.database[name]  # pylint: disable=pointless-statement

    def test__lazy_create_collection(self):
        col = self.database.a
        self.assertEqual(set(self.database.list_collection_names()), set())
        col.insert_one({"foo": "bar"})
        self.assertEqual(set(self.database.list_collection_names()), set(["a"]))

    def test__equality(self):
        self.assertEqual(self.database, self.database)
        client = mongomock.MongoClient("localhost")
        self.assertNotEqual(client.a, client.b)
        self.assertEqual(client.a, client.get_database("a"))
        self.assertEqual(client.a, mongomock.MongoClient("localhost").a)
        self.assertNotEqual(client.a, mongomock.MongoClient("example.com").a)

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
            {self.database}  # pylint: disable=pointless-statement

    @skipIf(
        sys.version_info < (3,),
        "Older versions of Python do not handle hashing the same way",
    )
    @skipIf(
        helpers.PYMONGO_VERSION < version.parse("3.12"),
        "older versions of pymongo didn't have proper hashing",
    )
    def test__hashable(self):
        {self.database}  # pylint: disable=pointless-statement

    def test__bad_type_as_a_read_concern_returns_type_error(self):
        client = mongomock.MongoClient()
        with self.assertRaises(
            TypeError,
            msg="read_concern must be an instance of pymongo.read_concern.ReadConcern",
        ):
            mongomock.database.Database(client, "foo", None, read_concern="bar")


_DBRef = collections.namedtuple("DBRef", ["database", "collection", "id"])
