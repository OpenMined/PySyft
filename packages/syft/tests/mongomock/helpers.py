# stdlib
from collections import OrderedDict
from collections import abc
from datetime import datetime
from datetime import timedelta
from datetime import tzinfo
import re
import time
from urllib.parse import unquote_plus
import warnings

# third party
from packaging import version

# relative
from . import InvalidURI

# Get ObjectId from bson if available or import a crafted one. This is not used
# in this module but is made available for callers of this module.
try:
    # third party
    from bson import ObjectId  # pylint: disable=unused-import
    from bson import Timestamp
    from pymongo import version as pymongo_version

    PYMONGO_VERSION = version.parse(pymongo_version)
    HAVE_PYMONGO = True
except ImportError:
    from .object_id import ObjectId  # noqa

    Timestamp = None
    # Default Pymongo version if not present.
    PYMONGO_VERSION = version.parse("4.0")
    HAVE_PYMONGO = False

# Cache the RegExp pattern type.
RE_TYPE = type(re.compile(""))
_HOST_MATCH = re.compile(r"^([^@]+@)?([^:]+|\[[^\]]+\])(:([^:]+))?$")
_SIMPLE_HOST_MATCH = re.compile(r"^([^:]+|\[[^\]]+\])(:([^:]+))?$")

try:
    # third party
    from bson.tz_util import utc
except ImportError:

    class _FixedOffset(tzinfo):
        def __init__(self, offset, name):
            self.__offset = timedelta(minutes=offset)
            self.__name = name

        def __getinitargs__(self):
            return self.__offset, self.__name

        def utcoffset(self, dt):
            return self.__offset

        def tzname(self, dt):
            return self.__name

        def dst(self, dt):
            return timedelta(0)

    utc = _FixedOffset(0, "UTC")


ASCENDING = 1
DESCENDING = -1


def utcnow():
    """Simple wrapper for datetime.utcnow

    This provides a centralized definition of "now" in the mongomock realm,
    allowing users to transform the value of "now" to the future or the past,
    based on their testing needs. For example:

    ```python
    def test_x(self):
        with mock.patch("mongomock.utcnow") as mm_utc:
            mm_utc = datetime.utcnow() + timedelta(hours=100)
            # Test some things "100 hours" in the future
    ```
    """
    return datetime.utcnow()


def print_deprecation_warning(old_param_name, new_param_name):
    warnings.warn(
        "'%s' has been deprecated to be in line with pymongo implementation, a new parameter '%s' "
        "should be used instead. the old parameter will be kept for backward compatibility "
        "purposes." % (old_param_name, new_param_name),
        DeprecationWarning,
    )


def create_index_list(key_or_list, direction=None):
    """Helper to generate a list of (key, direction) pairs.

    It takes such a list, or a single key, or a single key and direction.
    """
    if isinstance(key_or_list, str):
        return [(key_or_list, direction or ASCENDING)]
    if not isinstance(key_or_list, (list, tuple, abc.Iterable)):
        raise TypeError(
            "if no direction is specified, " "key_or_list must be an instance of list"
        )
    return key_or_list


def gen_index_name(index_list):
    """Generate an index name based on the list of keys with directions."""

    return "_".join(["%s_%s" % item for item in index_list])


class hashdict(dict):
    """hashable dict implementation, suitable for use as a key into other dicts.

    >>> h1 = hashdict({'apples': 1, 'bananas':2})
    >>> h2 = hashdict({'bananas': 3, 'mangoes': 5})
    >>> h1+h2
    hashdict(apples=1, bananas=3, mangoes=5)
    >>> d1 = {}
    >>> d1[h1] = 'salad'
    >>> d1[h1]
    'salad'
    >>> d1[h2]
    Traceback (most recent call last):
    ...
    KeyError: hashdict(bananas=3, mangoes=5)

    based on answers from
    http://stackoverflow.com/questions/1151658/python-hashable-dicts
    """

    def __key(self):
        return frozenset(
            (
                k,
                (
                    hashdict(v)
                    if isinstance(v, dict)
                    else tuple(v)
                    if isinstance(v, list)
                    else v
                ),
            )
            for k, v in self.items()
        )

    def __repr__(self):
        return "{0}({1})".format(
            self.__class__.__name__,
            ", ".join(
                "{0}={1}".format(str(i[0]), repr(i[1])) for i in sorted(self.__key())
            ),
        )

    def __hash__(self):
        return hash(self.__key())

    def __setitem__(self, key, value):
        raise TypeError(
            "{0} does not support item assignment".format(self.__class__.__name__)
        )

    def __delitem__(self, key):
        raise TypeError(
            "{0} does not support item assignment".format(self.__class__.__name__)
        )

    def clear(self):
        raise TypeError(
            "{0} does not support item assignment".format(self.__class__.__name__)
        )

    def pop(self, *args, **kwargs):
        raise TypeError(
            "{0} does not support item assignment".format(self.__class__.__name__)
        )

    def popitem(self, *args, **kwargs):
        raise TypeError(
            "{0} does not support item assignment".format(self.__class__.__name__)
        )

    def setdefault(self, *args, **kwargs):
        raise TypeError(
            "{0} does not support item assignment".format(self.__class__.__name__)
        )

    def update(self, *args, **kwargs):
        raise TypeError(
            "{0} does not support item assignment".format(self.__class__.__name__)
        )

    def __add__(self, right):
        result = hashdict(self)
        dict.update(result, right)
        return result


def fields_list_to_dict(fields):
    """Takes a list of field names and returns a matching dictionary.

    ['a', 'b'] becomes {'a': 1, 'b': 1}

    and

    ['a.b.c', 'd', 'a.c'] becomes {'a.b.c': 1, 'd': 1, 'a.c': 1}
    """
    as_dict = {}
    for field in fields:
        if not isinstance(field, str):
            raise TypeError(
                "fields must be a list of key names, each an instance of str"
            )
        as_dict[field] = 1
    return as_dict


def parse_uri(uri, default_port=27017, warn=False):
    """A simplified version of pymongo.uri_parser.parse_uri.

    Returns a dict with:
     - nodelist, a tuple of (host, port)
     - database the name of the database or None if no database is provided in the URI.

    An invalid MongoDB connection URI may raise an InvalidURI exception,
    however, the URI is not fully parsed and some invalid URIs may not result
    in an exception.

    'mongodb://host1/database' becomes 'host1', 27017, 'database'

    and

    'mongodb://host1' becomes 'host1', 27017, None
    """
    SCHEME = "mongodb://"

    if not uri.startswith(SCHEME):
        raise InvalidURI("Invalid URI scheme: URI " "must begin with '%s'" % (SCHEME,))

    scheme_free = uri[len(SCHEME) :]

    if not scheme_free:
        raise InvalidURI("Must provide at least one hostname or IP.")

    dbase = None

    # Check for unix domain sockets in the uri
    if ".sock" in scheme_free:
        host_part, _, path_part = scheme_free.rpartition("/")
        if not host_part:
            host_part = path_part
            path_part = ""
        if "/" in host_part:
            raise InvalidURI(
                "Any '/' in a unix domain socket must be" " URL encoded: %s" % host_part
            )
        path_part = unquote_plus(path_part)
    else:
        host_part, _, path_part = scheme_free.partition("/")

    if not path_part and "?" in host_part:
        raise InvalidURI("A '/' is required between " "the host list and any options.")

    nodelist = []
    if "," in host_part:
        hosts = host_part.split(",")
    else:
        hosts = [host_part]
    for host in hosts:
        match = _HOST_MATCH.match(host)
        if not match:
            raise ValueError(
                "Reserved characters such as ':' must be escaped according RFC "
                "2396. An IPv6 address literal must be enclosed in '[' and ']' "
                "according to RFC 2732."
            )
        host = match.group(2)
        if host.startswith("[") and host.endswith("]"):
            host = host[1:-1]

        port = match.group(4)
        if port:
            try:
                port = int(port)
                if port < 0 or port > 65535:
                    raise ValueError()
            except ValueError as err:
                raise ValueError(
                    "Port must be an integer between 0 and 65535:", port
                ) from err
        else:
            port = default_port

        nodelist.append((host, port))

    if path_part and path_part[0] != "?":
        dbase, _, _ = path_part.partition("?")
        if "." in dbase:
            dbase, _ = dbase.split(".", 1)

    if dbase is not None:
        dbase = unquote_plus(dbase)

    return {"nodelist": tuple(nodelist), "database": dbase}


def split_hosts(hosts, default_port=27017):
    """Split the entity into a list of tuples of host and port."""

    nodelist = []
    for entity in hosts.split(","):
        port = default_port
        if entity.endswith(".sock"):
            port = None

        match = _SIMPLE_HOST_MATCH.match(entity)
        if not match:
            raise ValueError(
                "Reserved characters such as ':' must be escaped according RFC "
                "2396. An IPv6 address literal must be enclosed in '[' and ']' "
                "according to RFC 2732."
            )
        host = match.group(1)
        if host.startswith("[") and host.endswith("]"):
            host = host[1:-1]

        if match.group(3):
            try:
                port = int(match.group(3))
                if port < 0 or port > 65535:
                    raise ValueError()
            except ValueError as err:
                raise ValueError(
                    "Port must be an integer between 0 and 65535:", port
                ) from err

        nodelist.append((host, port))

    return nodelist


_LAST_TIMESTAMP_INC = []


def get_current_timestamp():
    """Get the current timestamp as a bson Timestamp object."""
    if not Timestamp:
        raise NotImplementedError(
            "timestamp is not supported. Import pymongo to use it."
        )
    now = int(time.time())
    if _LAST_TIMESTAMP_INC and _LAST_TIMESTAMP_INC[0] == now:
        _LAST_TIMESTAMP_INC[1] += 1
    else:
        del _LAST_TIMESTAMP_INC[:]
        _LAST_TIMESTAMP_INC.extend([now, 1])
    return Timestamp(now, _LAST_TIMESTAMP_INC[1])


def patch_datetime_awareness_in_document(value):
    # MongoDB is supposed to stock everything as timezone naive utc date
    # Hence we have to convert incoming datetimes to avoid errors while
    # mixing tz aware and naive.
    # On top of that, MongoDB date precision is up to millisecond, where Python
    # datetime use microsecond, so we must lower the precision to mimic mongo.
    for best_type in (OrderedDict, dict):
        if isinstance(value, best_type):
            return best_type(
                (k, patch_datetime_awareness_in_document(v)) for k, v in value.items()
            )
    if isinstance(value, (tuple, list)):
        return [patch_datetime_awareness_in_document(item) for item in value]
    if isinstance(value, datetime):
        mongo_us = (value.microsecond // 1000) * 1000
        if value.tzinfo:
            return (value - value.utcoffset()).replace(
                tzinfo=None, microsecond=mongo_us
            )
        return value.replace(microsecond=mongo_us)
    if Timestamp and isinstance(value, Timestamp) and not value.time and not value.inc:
        return get_current_timestamp()
    return value


def make_datetime_timezone_aware_in_document(value):
    # MongoClient support tz_aware=True parameter to return timezone-aware
    # datetime objects. Given the date is stored internally without timezone
    # information, all returned datetime have utc as timezone.
    if isinstance(value, dict):
        return {
            k: make_datetime_timezone_aware_in_document(v) for k, v in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [make_datetime_timezone_aware_in_document(item) for item in value]
    if isinstance(value, datetime):
        return value.replace(tzinfo=utc)
    return value


def get_value_by_dot(doc, key, can_generate_array=False):
    """Get dictionary value using dotted key"""
    result = doc
    key_items = key.split(".")
    for key_index, key_item in enumerate(key_items):
        if isinstance(result, dict):
            result = result[key_item]

        elif isinstance(result, (list, tuple)):
            try:
                int_key = int(key_item)
            except ValueError as err:
                if not can_generate_array:
                    raise KeyError(key_index) from err
                remaining_key = ".".join(key_items[key_index:])
                return [get_value_by_dot(subdoc, remaining_key) for subdoc in result]

            try:
                result = result[int_key]
            except (ValueError, IndexError) as err:
                raise KeyError(key_index) from err

        else:
            raise KeyError(key_index)

    return result


def set_value_by_dot(doc, key, value):
    """Set dictionary value using dotted key"""
    try:
        parent_key, child_key = key.rsplit(".", 1)
        parent = get_value_by_dot(doc, parent_key)
    except ValueError:
        child_key = key
        parent = doc

    if isinstance(parent, dict):
        parent[child_key] = value
    elif isinstance(parent, (list, tuple)):
        try:
            parent[int(child_key)] = value
        except (ValueError, IndexError) as err:
            raise KeyError() from err
    else:
        raise KeyError()

    return doc


def delete_value_by_dot(doc, key):
    """Delete dictionary value using dotted key.

    This function assumes that the value exists.
    """
    try:
        parent_key, child_key = key.rsplit(".", 1)
        parent = get_value_by_dot(doc, parent_key)
    except ValueError:
        child_key = key
        parent = doc

    del parent[child_key]

    return doc


def mongodb_to_bool(value):
    """Converts any value to bool the way MongoDB does it"""

    return value not in [False, None, 0]
