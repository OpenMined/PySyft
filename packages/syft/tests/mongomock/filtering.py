# stdlib
from datetime import datetime
import itertools
import numbers
import operator
import re
import uuid

# relative
from . import OperationFailure
from .helpers import ObjectId
from .helpers import RE_TYPE

try:
    # stdlib
    from types import NoneType
except ImportError:
    NoneType = type(None)

try:
    # third party
    from bson import DBRef
    from bson import Regex

    _RE_TYPES = (RE_TYPE, Regex)
except ImportError:
    DBRef = None
    _RE_TYPES = (RE_TYPE,)

try:
    # third party
    from bson.decimal128 import Decimal128
except ImportError:
    Decimal128 = None

_TOP_LEVEL_OPERATORS = {"$expr", "$text", "$where", "$jsonSchema"}


_NOT_IMPLEMENTED_OPERATORS = {
    "$bitsAllClear",
    "$bitsAllSet",
    "$bitsAnyClear",
    "$bitsAnySet",
    "$geoIntersects",
    "$geoWithin",
    "$maxDistance",
    "$minDistance",
    "$near",
    "$nearSphere",
}


def filter_applies(search_filter, document):
    """Applies given filter

    This function implements MongoDB's matching strategy over documents in the find() method
    and other related scenarios (like $elemMatch)
    """
    return _filterer_inst.apply(search_filter, document)


class _Filterer(object):
    """An object to help applying a filter, using the MongoDB query language."""

    # This is populated using register_parse_expression further down.
    parse_expression = []

    def __init__(self):
        self._operator_map = dict(
            {
                "$eq": _list_expand(operator_eq),
                "$ne": _list_expand(
                    lambda dv, sv: not operator_eq(dv, sv), negative=True
                ),
                "$all": self._all_op,
                "$in": _in_op,
                "$nin": lambda dv, sv: not _in_op(dv, sv),
                "$exists": lambda dv, sv: bool(sv) == (dv is not None),
                "$regex": _not_None_and(_regex),
                "$elemMatch": self._elem_match_op,
                "$size": _size_op,
                "$type": _type_op,
            },
            **{
                key: _not_None_and(_list_expand(_compare_objects(op)))
                for key, op in SORTING_OPERATOR_MAP.items()
            },
        )

    def apply(self, search_filter, document):
        if not isinstance(search_filter, dict):
            raise OperationFailure(
                "the match filter must be an expression in an object"
            )

        for key, search in search_filter.items():
            # Top level operators.
            if key == "$comment":
                continue
            if key in LOGICAL_OPERATOR_MAP:
                if not search:
                    raise OperationFailure(
                        "BadValue $and/$or/$nor must be a nonempty array"
                    )
                if not LOGICAL_OPERATOR_MAP[key](document, search, self.apply):
                    return False
                continue
            if key == "$expr":
                parse_expression = self.parse_expression[0]
                if not parse_expression(search, document, ignore_missing_keys=True):
                    return False
                continue
            if key in _TOP_LEVEL_OPERATORS:
                raise NotImplementedError(
                    "The {} operator is not implemented in mongomock yet".format(key)
                )
            if key.startswith("$"):
                raise OperationFailure("unknown top level operator: " + key)

            is_match = False

            is_checking_negative_match = isinstance(search, dict) and {
                "$ne",
                "$nin",
            } & set(search.keys())
            is_checking_positive_match = not isinstance(search, dict) or (
                set(search.keys()) - {"$ne", "$nin"}
            )
            has_candidates = False

            if search == {"$exists": False} and not iter_key_candidates(key, document):
                continue

            if isinstance(search, dict) and "$all" in search:
                if not self._all_op(iter_key_candidates(key, document), search["$all"]):
                    return False
                # if there are no query operators then continue
                if len(search) == 1:
                    continue

            for doc_val in iter_key_candidates(key, document):
                has_candidates |= doc_val is not None
                is_ops_filter = (
                    search
                    and isinstance(search, dict)
                    and all(key.startswith("$") for key in search.keys())
                )
                if is_ops_filter:
                    if "$options" in search and "$regex" in search:
                        search = _combine_regex_options(search)
                    unknown_operators = set(search) - set(self._operator_map) - {"$not"}
                    if unknown_operators:
                        not_implemented_operators = (
                            unknown_operators & _NOT_IMPLEMENTED_OPERATORS
                        )
                        if not_implemented_operators:
                            raise NotImplementedError(
                                "'%s' is a valid operation but it is not supported by Mongomock "
                                "yet." % list(not_implemented_operators)[0]
                            )
                        raise OperationFailure(
                            "unknown operator: " + list(unknown_operators)[0]
                        )
                    is_match = (
                        all(
                            operator_string in self._operator_map
                            and self._operator_map[operator_string](doc_val, search_val)
                            or operator_string == "$not"
                            and self._not_op(document, key, search_val)
                            for operator_string, search_val in search.items()
                        )
                        and search
                    )
                elif isinstance(search, _RE_TYPES) and isinstance(doc_val, (str, list)):
                    is_match = _regex(doc_val, search)
                elif key in LOGICAL_OPERATOR_MAP:
                    if not search:
                        raise OperationFailure(
                            "BadValue $and/$or/$nor must be a nonempty array"
                        )
                    is_match = LOGICAL_OPERATOR_MAP[key](document, search, self.apply)
                elif isinstance(doc_val, (list, tuple)):
                    is_match = search in doc_val or search == doc_val
                    if isinstance(search, ObjectId):
                        is_match |= str(search) in doc_val
                else:
                    is_match = (doc_val == search) or (
                        search is None and doc_val is None
                    )

                # When checking negative match, all the elements should match.
                if is_checking_negative_match and not is_match:
                    return False

                # If not checking negative matches, the first match is enouh for this criteria.
                if is_match and not is_checking_negative_match:
                    break

            if not is_match and (has_candidates or is_checking_positive_match):
                return False

        return True

    def _not_op(self, d, k, s):
        if isinstance(s, dict):
            for key in s.keys():
                if key not in self._operator_map and key not in LOGICAL_OPERATOR_MAP:
                    raise OperationFailure("unknown operator: %s" % key)
        elif isinstance(s, _RE_TYPES):
            pass
        else:
            raise OperationFailure("$not needs a regex or a document")
        return not self.apply({k: s}, d)

    def _elem_match_op(self, doc_val, query):
        if not isinstance(doc_val, list):
            return False
        if not isinstance(query, dict):
            raise OperationFailure("$elemMatch needs an Object")
        for item in doc_val:
            try:
                if self.apply(query, item):
                    return True
            except OperationFailure:
                if self.apply({"field": query}, {"field": item}):
                    return True
        return False

    def _all_op(self, doc_val, search_val):
        if isinstance(doc_val, list) and doc_val and isinstance(doc_val[0], list):
            doc_val = list(itertools.chain.from_iterable(doc_val))
        dv = _force_list(doc_val)
        matches = []
        for x in search_val:
            if isinstance(x, dict) and "$elemMatch" in x:
                matches.append(self._elem_match_op(doc_val, x["$elemMatch"]))
            else:
                matches.append(x in dv)
        return all(matches)


def iter_key_candidates(key, doc):
    """Get possible subdocuments or lists that are referred to by the key in question

    Returns the appropriate nested value if the key includes dot notation.
    """
    if not key:
        return [doc]

    if doc is None:
        return ()

    if isinstance(doc, list):
        return _iter_key_candidates_sublist(key, doc)

    if not isinstance(doc, dict):
        return ()

    key_parts = key.split(".")
    if len(key_parts) == 1:
        return [doc.get(key, None)]

    sub_key = ".".join(key_parts[1:])
    sub_doc = doc.get(key_parts[0], {})
    return iter_key_candidates(sub_key, sub_doc)


def _iter_key_candidates_sublist(key, doc):
    """Iterates of candidates

    :param doc: a list to be searched for candidates for our key
    :param key: the string key to be matched
    """
    key_parts = key.split(".")
    sub_key = key_parts.pop(0)
    key_remainder = ".".join(key_parts)
    try:
        sub_key_int = int(sub_key)
    except ValueError:
        sub_key_int = None

    if sub_key_int is None:
        # subkey is not an integer...
        ret = []
        for sub_doc in doc:
            if isinstance(sub_doc, dict):
                if sub_key in sub_doc:
                    ret.extend(iter_key_candidates(key_remainder, sub_doc[sub_key]))
                else:
                    ret.append(None)
        return ret

    # subkey is an index
    if sub_key_int >= len(doc):
        return ()  # dead end
    sub_doc = doc[sub_key_int]
    if key_parts:
        return iter_key_candidates(".".join(key_parts), sub_doc)
    return [sub_doc]


def _force_list(v):
    return v if isinstance(v, (list, tuple)) else [v]


def _in_op(doc_val, search_val):
    if not isinstance(search_val, (list, tuple)):
        raise OperationFailure("$in needs an array")
    if doc_val is None and None in search_val:
        return True
    doc_val = _force_list(doc_val)
    is_regex_list = [isinstance(x, _RE_TYPES) for x in search_val]
    if not any(is_regex_list):
        return any(x in search_val for x in doc_val)
    for x, is_regex in zip(search_val, is_regex_list):
        if (is_regex and _regex(doc_val, x)) or (x in doc_val):
            return True
    return False


def _not_None_and(f):
    """wrap an operator to return False if the first arg is None"""
    return lambda v, l: v is not None and f(v, l)


def _compare_objects(op):
    """Wrap an operator to also compare objects following BSON comparison.

    See https://docs.mongodb.com/manual/reference/bson-type-comparison-order/#objects
    """

    def _wrapped(a, b):
        # Do not compare uncomparable types, see Type Bracketing:
        # https://docs.mongodb.com/manual/reference/method/db.collection.find/#type-bracketing
        return bson_compare(op, a, b, can_compare_types=False)

    return _wrapped


def bson_compare(op, a, b, can_compare_types=True):
    """Compare two elements using BSON comparison.

    Args:
        op: the basic operation to compare (e.g. operator.lt, operator.ge).
        a: the first operand
        b: the second operand
        can_compare_types: if True, according to BSON's definition order
            between types is used, otherwise always return False when types are
            different.
    """
    a_type = _get_compare_type(a)
    b_type = _get_compare_type(b)
    if a_type != b_type:
        return can_compare_types and op(a_type, b_type)

    # Compare DBRefs as dicts
    if type(a).__name__ == "DBRef" and hasattr(a, "as_doc"):
        a = a.as_doc()
    if type(b).__name__ == "DBRef" and hasattr(b, "as_doc"):
        b = b.as_doc()

    if isinstance(a, dict):
        # MongoDb server compares the type before comparing the keys
        # https://github.com/mongodb/mongo/blob/f10f214/src/mongo/bson/bsonelement.cpp#L516
        # even though the documentation does not say anything about that.
        a = [(_get_compare_type(v), k, v) for k, v in a.items()]
        b = [(_get_compare_type(v), k, v) for k, v in b.items()]

    if isinstance(a, (tuple, list)):
        for item_a, item_b in zip(a, b):
            if item_a != item_b:
                return bson_compare(op, item_a, item_b)
        return bson_compare(op, len(a), len(b))

    if isinstance(a, NoneType):
        return op(0, 0)

    # bson handles bytes as binary in python3+:
    # https://api.mongodb.com/python/current/api/bson/index.html
    if isinstance(a, bytes):
        # Performs the same operation as described by:
        # https://docs.mongodb.com/manual/reference/bson-type-comparison-order/#bindata
        if len(a) != len(b):
            return op(len(a), len(b))
        # bytes is always treated as subtype 0 by the bson library
    return op(a, b)


def _get_compare_type(val):
    """Get a number representing the base type of the value used for comparison.

    See https://docs.mongodb.com/manual/reference/bson-type-comparison-order/
    also https://github.com/mongodb/mongo/blob/46b28bb/src/mongo/bson/bsontypes.h#L175
    for canonical values.
    """
    if isinstance(val, NoneType):
        return 5
    if isinstance(val, bool):
        return 40
    if isinstance(val, numbers.Number):
        return 10
    if isinstance(val, str):
        return 15
    if isinstance(val, dict):
        return 20
    if isinstance(val, (tuple, list)):
        return 25
    if isinstance(val, uuid.UUID):
        return 30
    if isinstance(val, bytes):
        return 30
    if isinstance(val, ObjectId):
        return 35
    if isinstance(val, datetime):
        return 45
    if isinstance(val, _RE_TYPES):
        return 50
    if DBRef and isinstance(val, DBRef):
        # According to the C++ code, this should be 55 but apparently sending a DBRef through
        # pymongo is stored as a dict.
        return 20
    return 0


def _regex(doc_val, regex):
    if not (isinstance(doc_val, (str, list)) or isinstance(doc_val, RE_TYPE)):
        return False
    if isinstance(regex, str):
        regex = re.compile(regex)
    if not isinstance(regex, RE_TYPE):
        # bson.Regex
        regex = regex.try_compile()
    return any(
        regex.search(item) for item in _force_list(doc_val) if isinstance(item, str)
    )


def _size_op(doc_val, search_val):
    if isinstance(doc_val, (list, tuple, dict)):
        return search_val == len(doc_val)
    return search_val == 1 if doc_val and doc_val is not None else 0


def _list_expand(f, negative=False):
    def func(doc_val, search_val):
        if isinstance(doc_val, (list, tuple)) and not isinstance(
            search_val, (list, tuple)
        ):
            if negative:
                return all(f(val, search_val) for val in doc_val)
            return any(f(val, search_val) for val in doc_val)
        return f(doc_val, search_val)

    return func


def _type_op(doc_val, search_val, in_array=False):
    if search_val not in TYPE_MAP:
        raise OperationFailure("%r is not a valid $type" % search_val)
    elif TYPE_MAP[search_val] is None:
        raise NotImplementedError(
            "%s is a valid $type but not implemented" % search_val
        )
    if TYPE_MAP[search_val](doc_val):
        return True
    if isinstance(doc_val, (list, tuple)) and not in_array:
        return any(_type_op(val, search_val, in_array=True) for val in doc_val)
    return False


def _combine_regex_options(search):
    if not isinstance(search["$options"], str):
        raise OperationFailure("$options has to be a string")

    options = None
    for option in search["$options"]:
        if option not in "imxs":
            continue
        re_option = getattr(re, option.upper())
        if options is None:
            options = re_option
        else:
            options |= re_option

    search_copy = dict(search)
    del search_copy["$options"]

    if options is None:
        return search_copy

    if isinstance(search["$regex"], _RE_TYPES):
        if isinstance(search["$regex"], RE_TYPE):
            search_copy["$regex"] = re.compile(
                search["$regex"].pattern, search["$regex"].flags | options
            )
        else:
            # bson.Regex
            regex = search["$regex"]
            search_copy["$regex"] = regex.__class__(
                regex.pattern, regex.flags | options
            )
    else:
        search_copy["$regex"] = re.compile(search["$regex"], options)
    return search_copy


def operator_eq(doc_val, search_val):
    if doc_val is None and search_val is None:
        return True
    return operator.eq(doc_val, search_val)


SORTING_OPERATOR_MAP = {
    "$gt": operator.gt,
    "$gte": operator.ge,
    "$lt": operator.lt,
    "$lte": operator.le,
}


LOGICAL_OPERATOR_MAP = {
    "$or": lambda d, subq, filter_func: any(filter_func(q, d) for q in subq),
    "$and": lambda d, subq, filter_func: all(filter_func(q, d) for q in subq),
    "$nor": lambda d, subq, filter_func: all(not filter_func(q, d) for q in subq),
    "$not": lambda d, subq, filter_func: (not filter_func(q, d) for q in subq),
}


TYPE_MAP = {
    "double": lambda v: isinstance(v, float),
    "string": lambda v: isinstance(v, str),
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "binData": lambda v: isinstance(v, bytes),
    "undefined": None,
    "objectId": lambda v: isinstance(v, ObjectId),
    "bool": lambda v: isinstance(v, bool),
    "date": lambda v: isinstance(v, datetime),
    "null": None,
    "regex": None,
    "dbPointer": None,
    "javascript": None,
    "symbol": None,
    "javascriptWithScope": None,
    "int": lambda v: (
        isinstance(v, int) and not isinstance(v, bool) and v.bit_length() <= 32
    ),
    "timestamp": None,
    "long": lambda v: (
        isinstance(v, int) and not isinstance(v, bool) and v.bit_length() > 32
    ),
    "decimal": (lambda v: isinstance(v, Decimal128)) if Decimal128 else None,
    "number": lambda v: (
        # pylint: disable-next=isinstance-second-argument-not-valid-type
        isinstance(v, (int, float) + ((Decimal128,) if Decimal128 else ()))
        and not isinstance(v, bool)
    ),
    "minKey": None,
    "maxKey": None,
}


def resolve_key(key, doc):
    return next(iter(iter_key_candidates(key, doc)), None)


def resolve_sort_key(key, doc):
    value = resolve_key(key, doc)
    # see http://docs.mongodb.org/manual/reference/method/cursor.sort/#ascending-descending-sort
    if value is None:
        return 1, BsonComparable(None)

    # List or tuples are sorted solely by their first value.
    if isinstance(value, (tuple, list)):
        if not value:
            return 0, BsonComparable(None)
        return 1, BsonComparable(value[0])

    return 1, BsonComparable(value)


class BsonComparable(object):
    """Wraps a value in an BSON like object that can be compared one to another."""

    def __init__(self, obj):
        self.obj = obj

    def __lt__(self, other):
        return bson_compare(operator.lt, self.obj, other.obj)


_filterer_inst = _Filterer()


# Developer note: to avoid a cross-modules dependency (filtering requires aggregation, that requires
# filtering), the aggregation module needs to register its parse_expression function here.
def register_parse_expression(parse_expression):
    """Register the parse_expression function from the aggregate module."""

    del _Filterer.parse_expression[:]
    _Filterer.parse_expression.append(parse_expression)
