"""Module to handle the operations within the aggregate pipeline."""

# stdlib
import bisect
import collections
import copy
import datetime
import decimal
import functools
import itertools
import math
import numbers
import random
import re
import sys
import warnings

# third party
from packaging import version
import pytz

# relative
from . import OperationFailure
from . import command_cursor
from . import filtering
from . import helpers

try:
    # third party
    from bson import Regex
    from bson import decimal128
    from bson.errors import InvalidDocument

    decimal_support = True
    _RE_TYPES = (helpers.RE_TYPE, Regex)
except ImportError:
    InvalidDocument = OperationFailure
    decimal_support = False
    _RE_TYPES = helpers.RE_TYPE

_random = random.Random()


group_operators = [
    "$addToSet",
    "$avg",
    "$first",
    "$last",
    "$max",
    "$mergeObjects",
    "$min",
    "$push",
    "$stdDevPop",
    "$stdDevSamp",
    "$sum",
]
unary_arithmetic_operators = {
    "$abs",
    "$ceil",
    "$exp",
    "$floor",
    "$ln",
    "$log10",
    "$sqrt",
    "$trunc",
}
binary_arithmetic_operators = {
    "$divide",
    "$log",
    "$mod",
    "$pow",
    "$subtract",
}
arithmetic_operators = (
    unary_arithmetic_operators
    | binary_arithmetic_operators
    | {
        "$add",
        "$multiply",
    }
)
project_operators = [
    "$max",
    "$min",
    "$avg",
    "$sum",
    "$stdDevPop",
    "$stdDevSamp",
    "$arrayElemAt",
    "$first",
    "$last",
]
control_flow_operators = [
    "$switch",
]
projection_operators = [
    "$let",
    "$literal",
]
date_operators = [
    "$dateFromString",
    "$dateToString",
    "$dateFromParts",
    "$dayOfMonth",
    "$dayOfWeek",
    "$dayOfYear",
    "$hour",
    "$isoDayOfWeek",
    "$isoWeek",
    "$isoWeekYear",
    "$millisecond",
    "$minute",
    "$month",
    "$second",
    "$week",
    "$year",
]
conditional_operators = ["$cond", "$ifNull"]
array_operators = [
    "$concatArrays",
    "$filter",
    "$indexOfArray",
    "$map",
    "$range",
    "$reduce",
    "$reverseArray",
    "$size",
    "$slice",
    "$zip",
]
object_operators = [
    "$mergeObjects",
]
text_search_operators = ["$meta"]
string_operators = [
    "$concat",
    "$indexOfBytes",
    "$indexOfCP",
    "$regexMatch",
    "$split",
    "$strcasecmp",
    "$strLenBytes",
    "$strLenCP",
    "$substr",
    "$substrBytes",
    "$substrCP",
    "$toLower",
    "$toUpper",
    "$trim",
]
comparison_operators = [
    "$cmp",
    "$eq",
    "$ne",
] + list(filtering.SORTING_OPERATOR_MAP.keys())
boolean_operators = ["$and", "$or", "$not"]
set_operators = [
    "$in",
    "$setEquals",
    "$setIntersection",
    "$setDifference",
    "$setUnion",
    "$setIsSubset",
    "$anyElementTrue",
    "$allElementsTrue",
]

type_convertion_operators = [
    "$convert",
    "$toString",
    "$toInt",
    "$toDecimal",
    "$toLong",
    "$arrayToObject",
    "$objectToArray",
]
type_operators = [
    "$isNumber",
    "$isArray",
]


def _avg_operation(values):
    values_list = list(v for v in values if isinstance(v, numbers.Number))
    if not values_list:
        return None
    return sum(values_list) / float(len(list(values_list)))


def _group_operation(values, operator):
    values_list = list(v for v in values if v is not None)
    if not values_list:
        return None
    return operator(values_list)


def _sum_operation(values):
    values_list = list()
    if decimal_support:
        for v in values:
            if isinstance(v, numbers.Number):
                values_list.append(v)
            elif isinstance(v, decimal128.Decimal128):
                values_list.append(v.to_decimal())
    else:
        values_list = list(v for v in values if isinstance(v, numbers.Number))
    sum_value = sum(values_list)
    return (
        decimal128.Decimal128(sum_value)
        if isinstance(sum_value, decimal.Decimal)
        else sum_value
    )


def _merge_objects_operation(values):
    merged_doc = dict()
    for v in values:
        if isinstance(v, dict):
            merged_doc.update(v)
    return merged_doc


_GROUPING_OPERATOR_MAP = {
    "$sum": _sum_operation,
    "$avg": _avg_operation,
    "$mergeObjects": _merge_objects_operation,
    "$min": lambda values: _group_operation(values, min),
    "$max": lambda values: _group_operation(values, max),
    "$first": lambda values: values[0] if values else None,
    "$last": lambda values: values[-1] if values else None,
}


class _Parser(object):
    """Helper to parse expressions within the aggregate pipeline."""

    def __init__(self, doc_dict, user_vars=None, ignore_missing_keys=False):
        self._doc_dict = doc_dict
        self._ignore_missing_keys = ignore_missing_keys
        self._user_vars = user_vars or {}

    def parse(self, expression):
        """Parse a MongoDB expression."""
        if not isinstance(expression, dict):
            # May raise a KeyError despite the ignore missing key.
            return self._parse_basic_expression(expression)

        if len(expression) > 1 and any(key.startswith("$") for key in expression):
            raise OperationFailure(
                "an expression specification must contain exactly one field, "
                "the name of the expression. Found %d fields in %s"
                % (len(expression), expression)
            )

        value_dict = {}
        for k, v in expression.items():
            if k in arithmetic_operators:
                return self._handle_arithmetic_operator(k, v)
            if k in project_operators:
                return self._handle_project_operator(k, v)
            if k in projection_operators:
                return self._handle_projection_operator(k, v)
            if k in comparison_operators:
                return self._handle_comparison_operator(k, v)
            if k in date_operators:
                return self._handle_date_operator(k, v)
            if k in array_operators:
                return self._handle_array_operator(k, v)
            if k in conditional_operators:
                return self._handle_conditional_operator(k, v)
            if k in control_flow_operators:
                return self._handle_control_flow_operator(k, v)
            if k in set_operators:
                return self._handle_set_operator(k, v)
            if k in string_operators:
                return self._handle_string_operator(k, v)
            if k in type_convertion_operators:
                return self._handle_type_convertion_operator(k, v)
            if k in type_operators:
                return self._handle_type_operator(k, v)
            if k in boolean_operators:
                return self._handle_boolean_operator(k, v)
            if k in text_search_operators + projection_operators + object_operators:
                raise NotImplementedError(
                    "'%s' is a valid operation but it is not supported by Mongomock yet."
                    % k
                )
            if k.startswith("$"):
                raise OperationFailure("Unrecognized expression '%s'" % k)
            try:
                value = self.parse(v)
            except KeyError:
                if self._ignore_missing_keys:
                    continue
                raise
            value_dict[k] = value

        return value_dict

    def parse_many(self, values):
        for value in values:
            try:
                yield self.parse(value)
            except KeyError:
                if self._ignore_missing_keys:
                    yield None
                else:
                    raise

    def _parse_to_bool(self, expression):
        """Parse a MongoDB expression and then convert it to bool"""
        # handles converting `undefined` (in form of KeyError) to False
        try:
            return helpers.mongodb_to_bool(self.parse(expression))
        except KeyError:
            return False

    def _parse_or_None(self, expression):
        try:
            return self.parse(expression)
        except KeyError:
            return None

    def _parse_basic_expression(self, expression):
        if isinstance(expression, str) and expression.startswith("$"):
            if expression.startswith("$$"):
                return helpers.get_value_by_dot(
                    dict(
                        {
                            "ROOT": self._doc_dict,
                            "CURRENT": self._doc_dict,
                        },
                        **self._user_vars,
                    ),
                    expression[2:],
                    can_generate_array=True,
                )
            return helpers.get_value_by_dot(
                self._doc_dict, expression[1:], can_generate_array=True
            )
        return expression

    def _handle_boolean_operator(self, operator, values):
        if operator == "$and":
            return all([self._parse_to_bool(value) for value in values])
        if operator == "$or":
            return any(self._parse_to_bool(value) for value in values)
        if operator == "$not":
            return not self._parse_to_bool(values)
        # This should never happen: it is only a safe fallback if something went wrong.
        raise NotImplementedError(  # pragma: no cover
            "Although '%s' is a valid boolean operator for the "
            "aggregation pipeline, it is currently not implemented"
            " in Mongomock." % operator
        )

    def _handle_arithmetic_operator(self, operator, values):
        if operator in unary_arithmetic_operators:
            try:
                number = self.parse(values)
            except KeyError:
                return None
            if number is None:
                return None
            if not isinstance(number, numbers.Number):
                raise OperationFailure(
                    "Parameter to %s must evaluate to a number, got '%s'"
                    % (operator, type(number))
                )

            if operator == "$abs":
                return abs(number)
            if operator == "$ceil":
                return math.ceil(number)
            if operator == "$exp":
                return math.exp(number)
            if operator == "$floor":
                return math.floor(number)
            if operator == "$ln":
                return math.log(number)
            if operator == "$log10":
                return math.log10(number)
            if operator == "$sqrt":
                return math.sqrt(number)
            if operator == "$trunc":
                return math.trunc(number)

        if operator in binary_arithmetic_operators:
            if not isinstance(values, (tuple, list)):
                raise OperationFailure(
                    "Parameter to %s must evaluate to a list, got '%s'"
                    % (operator, type(values))
                )

            if len(values) != 2:
                raise OperationFailure("%s must have only 2 parameters" % operator)
            number_0, number_1 = self.parse_many(values)
            if number_0 is None or number_1 is None:
                return None

            if operator == "$divide":
                return number_0 / number_1
            if operator == "$log":
                return math.log(number_0, number_1)
            if operator == "$mod":
                return math.fmod(number_0, number_1)
            if operator == "$pow":
                return math.pow(number_0, number_1)
            if operator == "$subtract":
                if isinstance(number_0, datetime.datetime) and isinstance(
                    number_1, (int, float)
                ):
                    number_1 = datetime.timedelta(milliseconds=number_1)
                res = number_0 - number_1
                if isinstance(res, datetime.timedelta):
                    return round(res.total_seconds() * 1000)
                return res

        assert isinstance(values, (tuple, list)), (
            "Parameter to %s must evaluate to a list, got '%s'"
            % (
                operator,
                type(values),
            )
        )

        parsed_values = list(self.parse_many(values))
        assert parsed_values, "%s must have at least one parameter" % operator
        for value in parsed_values:
            if value is None:
                return None
            assert isinstance(value, numbers.Number), "%s only uses numbers" % operator
        if operator == "$add":
            return sum(parsed_values)
        if operator == "$multiply":
            return functools.reduce(lambda x, y: x * y, parsed_values)

        # This should never happen: it is only a safe fallback if something went wrong.
        raise NotImplementedError(  # pragma: no cover
            "Although '%s' is a valid aritmetic operator for the aggregation "
            "pipeline, it is currently not implemented  in Mongomock." % operator
        )

    def _handle_project_operator(self, operator, values):
        if operator in _GROUPING_OPERATOR_MAP:
            values = (
                self.parse(values)
                if isinstance(values, str)
                else self.parse_many(values)
            )
            return _GROUPING_OPERATOR_MAP[operator](values)
        if operator == "$arrayElemAt":
            key, value = values
            array = self.parse(key)
            index = self.parse(value)
            try:
                return array[index]
            except IndexError as error:
                raise KeyError("Array have length less than index value") from error

        raise NotImplementedError(
            "Although '%s' is a valid project operator for the "
            "aggregation pipeline, it is currently not implemented "
            "in Mongomock." % operator
        )

    def _handle_projection_operator(self, operator, value):
        if operator == "$literal":
            return value
        if operator == "$let":
            if not isinstance(value, dict):
                raise InvalidDocument("$let only supports an object as its argument")
            for field in ("vars", "in"):
                if field not in value:
                    raise OperationFailure(
                        "Missing '{}' parameter to $let".format(field)
                    )
            if not isinstance(value["vars"], dict):
                raise OperationFailure("invalid parameter: expected an object (vars)")
            user_vars = {
                var_key: self.parse(var_value)
                for var_key, var_value in value["vars"].items()
            }
            return _Parser(
                self._doc_dict,
                dict(self._user_vars, **user_vars),
                ignore_missing_keys=self._ignore_missing_keys,
            ).parse(value["in"])
        raise NotImplementedError(
            "Although '%s' is a valid project operator for the "
            "aggregation pipeline, it is currently not implemented "
            "in Mongomock." % operator
        )

    def _handle_comparison_operator(self, operator, values):
        assert len(values) == 2, "Comparison requires two expressions"
        a = self.parse(values[0])
        b = self.parse(values[1])
        if operator == "$eq":
            return a == b
        if operator == "$ne":
            return a != b
        if operator in filtering.SORTING_OPERATOR_MAP:
            return filtering.bson_compare(
                filtering.SORTING_OPERATOR_MAP[operator], a, b
            )
        raise NotImplementedError(
            "Although '%s' is a valid comparison operator for the "
            "aggregation pipeline, it is currently not implemented "
            " in Mongomock." % operator
        )

    def _handle_string_operator(self, operator, values):
        if operator == "$toLower":
            parsed = self.parse(values)
            return str(parsed).lower() if parsed is not None else ""
        if operator == "$toUpper":
            parsed = self.parse(values)
            return str(parsed).upper() if parsed is not None else ""
        if operator == "$concat":
            parsed_list = list(self.parse_many(values))
            return (
                None if None in parsed_list else "".join([str(x) for x in parsed_list])
            )
        if operator == "$split":
            if len(values) != 2:
                raise OperationFailure("split must have 2 items")
            try:
                string = self.parse(values[0])
                delimiter = self.parse(values[1])
            except KeyError:
                return None

            if string is None or delimiter is None:
                return None
            if not isinstance(string, str):
                raise TypeError("split first argument must evaluate to string")
            if not isinstance(delimiter, str):
                raise TypeError("split second argument must evaluate to string")
            return string.split(delimiter)
        if operator == "$substr":
            if len(values) != 3:
                raise OperationFailure("substr must have 3 items")
            string = str(self.parse(values[0]))
            first = self.parse(values[1])
            length = self.parse(values[2])
            if string is None:
                return ""
            if first < 0:
                warnings.warn(
                    "Negative starting point given to $substr is accepted only until "
                    "MongoDB 3.7. This behavior will change in the future."
                )
                return ""
            if length < 0:
                warnings.warn(
                    "Negative length given to $substr is accepted only until "
                    "MongoDB 3.7. This behavior will change in the future."
                )
            second = len(string) if length < 0 else first + length
            return string[first:second]
        if operator == "$strcasecmp":
            if len(values) != 2:
                raise OperationFailure("strcasecmp must have 2 items")
            a, b = str(self.parse(values[0])), str(self.parse(values[1]))
            return 0 if a == b else -1 if a < b else 1
        if operator == "$regexMatch":
            if not isinstance(values, dict):
                raise OperationFailure(
                    "$regexMatch expects an object of named arguments but found: %s"
                    % type(values)
                )
            for field in ("input", "regex"):
                if field not in values:
                    raise OperationFailure(
                        "$regexMatch requires '%s' parameter" % field
                    )
            unknown_args = set(values) - {"input", "regex", "options"}
            if unknown_args:
                raise OperationFailure(
                    "$regexMatch found an unknown argument: %s" % list(unknown_args)[0]
                )

            try:
                input_value = self.parse(values["input"])
            except KeyError:
                return False
            if not isinstance(input_value, str):
                raise OperationFailure("$regexMatch needs 'input' to be of type string")

            try:
                regex_val = self.parse(values["regex"])
            except KeyError:
                return False
            options = None
            for option in values.get("options", ""):
                if option not in "imxs":
                    raise OperationFailure(
                        "$regexMatch invalid flag in regex options: %s" % option
                    )
                re_option = getattr(re, option.upper())
                if options is None:
                    options = re_option
                else:
                    options |= re_option
            if isinstance(regex_val, str):
                if options is None:
                    regex = re.compile(regex_val)
                else:
                    regex = re.compile(regex_val, options)
            elif "options" in values and regex_val.flags:
                raise OperationFailure(
                    "$regexMatch: regex option(s) specified in both 'regex' and 'option' fields"
                )
            elif isinstance(regex_val, helpers.RE_TYPE):
                if options and not regex_val.flags:
                    regex = re.compile(regex_val.pattern, options)
                elif regex_val.flags & ~(re.I | re.M | re.X | re.S):
                    raise OperationFailure(
                        "$regexMatch invalid flag in regex options: %s"
                        % regex_val.flags
                    )
                else:
                    regex = regex_val
            elif isinstance(regex_val, _RE_TYPES):
                # bson.Regex
                if regex_val.flags & ~(re.I | re.M | re.X | re.S):
                    raise OperationFailure(
                        "$regexMatch invalid flag in regex options: %s"
                        % regex_val.flags
                    )
                regex = re.compile(regex_val.pattern, regex_val.flags or options)
            else:
                raise OperationFailure(
                    "$regexMatch needs 'regex' to be of type string or regex"
                )

            return bool(regex.search(input_value))

        # This should never happen: it is only a safe fallback if something went wrong.
        raise NotImplementedError(  # pragma: no cover
            "Although '%s' is a valid string operator for the aggregation "
            "pipeline, it is currently not implemented  in Mongomock." % operator
        )

    def _handle_date_operator(self, operator, values):
        if isinstance(values, dict) and values.keys() == {"date", "timezone"}:
            value = self.parse(values["date"])
            target_tz = pytz.timezone(values["timezone"])
            out_value = value.replace(tzinfo=pytz.utc).astimezone(target_tz)
        else:
            out_value = self.parse(values)

        if operator == "$dayOfYear":
            return out_value.timetuple().tm_yday
        if operator == "$dayOfMonth":
            return out_value.day
        if operator == "$dayOfWeek":
            return (out_value.isoweekday() % 7) + 1
        if operator == "$year":
            return out_value.year
        if operator == "$month":
            return out_value.month
        if operator == "$week":
            return int(out_value.strftime("%U"))
        if operator == "$hour":
            return out_value.hour
        if operator == "$minute":
            return out_value.minute
        if operator == "$second":
            return out_value.second
        if operator == "$millisecond":
            return int(out_value.microsecond / 1000)
        if operator == "$dateToString":
            if not isinstance(values, dict):
                raise OperationFailure(
                    "$dateToString operator must correspond a dict"
                    'that has "format" and "date" field.'
                )
            if not isinstance(values, dict) or not {"format", "date"} <= set(values):
                raise OperationFailure(
                    "$dateToString operator must correspond a dict"
                    'that has "format" and "date" field.'
                )
            if "%L" in out_value["format"]:
                raise NotImplementedError(
                    "Although %L is a valid date format for the "
                    "$dateToString operator, it is currently not implemented "
                    " in Mongomock."
                )
            if "onNull" in values:
                raise NotImplementedError(
                    "Although onNull is a valid field for the "
                    "$dateToString operator, it is currently not implemented "
                    " in Mongomock."
                )
            if "timezone" in values.keys():
                raise NotImplementedError(
                    "Although timezone is a valid field for the "
                    "$dateToString operator, it is currently not implemented "
                    " in Mongomock."
                )
            return out_value["date"].strftime(out_value["format"])
        if operator == "$dateFromParts":
            if not isinstance(out_value, dict):
                raise OperationFailure(
                    f"{operator} operator must correspond a dict "
                    'that has "year" or "isoWeekYear" field.'
                )
            if len(set(out_value) & {"year", "isoWeekYear"}) != 1:
                raise OperationFailure(
                    f"{operator} operator must correspond a dict "
                    'that has "year" or "isoWeekYear" field.'
                )
            for field in ("isoWeekYear", "isoWeek", "isoDayOfWeek", "timezone"):
                if field in out_value:
                    raise NotImplementedError(
                        f"Although {field} is a valid field for the "
                        f"{operator} operator, it is currently not implemented "
                        "in Mongomock."
                    )

            year = out_value["year"]
            month = out_value.get("month", 1) or 1
            day = out_value.get("day", 1) or 1
            hour = out_value.get("hour", 0) or 0
            minute = out_value.get("minute", 0) or 0
            second = out_value.get("second", 0) or 0
            millisecond = out_value.get("millisecond", 0) or 0

            return datetime.datetime(
                year=year,
                month=month,
                day=day,
                hour=hour,
                minute=minute,
                second=second,
                microsecond=millisecond,
            )

        raise NotImplementedError(
            "Although '%s' is a valid date operator for the "
            "aggregation pipeline, it is currently not implemented "
            " in Mongomock." % operator
        )

    def _handle_array_operator(self, operator, value):
        if operator == "$concatArrays":
            if not isinstance(value, (list, tuple)):
                value = [value]

            parsed_list = list(self.parse_many(value))
            for parsed_item in parsed_list:
                if parsed_item is not None and not isinstance(
                    parsed_item, (list, tuple)
                ):
                    raise OperationFailure(
                        "$concatArrays only supports arrays, not {}".format(
                            type(parsed_item)
                        )
                    )

            return (
                None
                if None in parsed_list
                else list(itertools.chain.from_iterable(parsed_list))
            )

        if operator == "$map":
            if not isinstance(value, dict):
                raise OperationFailure("$map only supports an object as its argument")

            # NOTE: while the two validations below could be achieved with
            # one-liner set operations (e.g. set(value) - {'input', 'as',
            # 'in'}), we prefer the iteration-based approaches in order to
            # mimic MongoDB's behavior regarding the order of evaluation. For
            # example, MongoDB complains about 'input' parameter missing before
            # 'in'.
            for k in ("input", "in"):
                if k not in value:
                    raise OperationFailure("Missing '%s' parameter to $map" % k)

            for k in value:
                if k not in {"input", "as", "in"}:
                    raise OperationFailure("Unrecognized parameter to $map: %s" % k)

            input_array = self._parse_or_None(value["input"])

            if input_array is None or input_array is None:
                return None

            if not isinstance(input_array, (list, tuple)):
                raise OperationFailure(
                    "input to $map must be an array not %s" % type(input_array)
                )

            fieldname = value.get("as", "this")
            in_expr = value["in"]
            return [
                _Parser(
                    self._doc_dict,
                    dict(self._user_vars, **{fieldname: item}),
                    ignore_missing_keys=self._ignore_missing_keys,
                ).parse(in_expr)
                for item in input_array
            ]

        if operator == "$size":
            if isinstance(value, list):
                if len(value) != 1:
                    raise OperationFailure(
                        "Expression $size takes exactly 1 arguments. "
                        "%d were passed in." % len(value)
                    )
                value = value[0]
            array_value = self._parse_or_None(value)
            if not isinstance(array_value, (list, tuple)):
                raise OperationFailure(
                    "The argument to $size must be an array, but was of type: %s"
                    % ("missing" if array_value is None else type(array_value))
                )
            return len(array_value)

        if operator == "$filter":
            if not isinstance(value, dict):
                raise OperationFailure(
                    "$filter only supports an object as its argument"
                )
            extra_params = set(value) - {"input", "cond", "as"}
            if extra_params:
                raise OperationFailure(
                    "Unrecognized parameter to $filter: %s" % extra_params.pop()
                )
            missing_params = {"input", "cond"} - set(value)
            if missing_params:
                raise OperationFailure(
                    "Missing '%s' parameter to $filter" % missing_params.pop()
                )

            input_array = self.parse(value["input"])
            fieldname = value.get("as", "this")
            cond = value["cond"]
            return [
                item
                for item in input_array
                if _Parser(
                    self._doc_dict,
                    dict(self._user_vars, **{fieldname: item}),
                    ignore_missing_keys=self._ignore_missing_keys,
                ).parse(cond)
            ]
        if operator == "$slice":
            if not isinstance(value, list):
                raise OperationFailure("$slice only supports a list as its argument")
            if len(value) < 2 or len(value) > 3:
                raise OperationFailure(
                    "Expression $slice takes at least 2 arguments, and at most "
                    "3, but {} were passed in".format(len(value))
                )
            array_value = self.parse(value[0])
            if not isinstance(array_value, list):
                raise OperationFailure(
                    "First argument to $slice must be an array, but is of type: {}".format(
                        type(array_value)
                    )
                )
            for num, v in zip(("Second", "Third"), value[1:]):
                if not isinstance(v, int):
                    raise OperationFailure(
                        "{} argument to $slice must be numeric, but is of type: {}".format(
                            num, type(v)
                        )
                    )
            if len(value) > 2 and value[2] <= 0:
                raise OperationFailure(
                    "Third argument to $slice must be " "positive: {}".format(value[2])
                )

            start = value[1]
            if start < 0:
                if len(value) > 2:
                    stop = len(array_value) + start + value[2]
                else:
                    stop = None
            elif len(value) > 2:
                stop = start + value[2]
            else:
                stop = start
                start = 0
            return array_value[start:stop]

        raise NotImplementedError(
            "Although '%s' is a valid array operator for the "
            "aggregation pipeline, it is currently not implemented "
            "in Mongomock." % operator
        )

    def _handle_type_convertion_operator(self, operator, values):
        if operator == "$toString":
            try:
                parsed = self.parse(values)
            except KeyError:
                return None
            if isinstance(parsed, bool):
                return str(parsed).lower()
            if isinstance(parsed, datetime.datetime):
                return parsed.isoformat()[:-3] + "Z"
            return str(parsed)

        if operator == "$toInt":
            try:
                parsed = self.parse(values)
            except KeyError:
                return None
            if decimal_support:
                if isinstance(parsed, decimal128.Decimal128):
                    return int(parsed.to_decimal())
                return int(parsed)
            raise NotImplementedError(
                "You need to import the pymongo library to support decimal128 type."
            )

        if operator == "$toLong":
            try:
                parsed = self.parse(values)
            except KeyError:
                return None
            if decimal_support:
                if isinstance(parsed, decimal128.Decimal128):
                    return int(parsed.to_decimal())
                return int(parsed)
            raise NotImplementedError(
                "You need to import the pymongo library to support decimal128 type."
            )

        # Document: https://docs.mongodb.com/manual/reference/operator/aggregation/toDecimal/
        if operator == "$toDecimal":
            if not decimal_support:
                raise NotImplementedError(
                    "You need to import the pymongo library to support decimal128 type."
                )
            try:
                parsed = self.parse(values)
            except KeyError:
                return None
            if isinstance(parsed, bool):
                parsed = "1" if parsed is True else "0"
                decimal_value = decimal128.Decimal128(parsed)
            elif isinstance(parsed, int):
                decimal_value = decimal128.Decimal128(str(parsed))
            elif isinstance(parsed, float):
                exp = decimal.Decimal(".00000000000000")
                decimal_value = decimal.Decimal(str(parsed)).quantize(exp)
                decimal_value = decimal128.Decimal128(decimal_value)
            elif isinstance(parsed, decimal128.Decimal128):
                decimal_value = parsed
            elif isinstance(parsed, str):
                try:
                    decimal_value = decimal128.Decimal128(parsed)
                except decimal.InvalidOperation as err:
                    raise OperationFailure(
                        "Failed to parse number '%s' in $convert with no onError value:"
                        "Failed to parse string to decimal" % parsed
                    ) from err
            elif isinstance(parsed, datetime.datetime):
                epoch = datetime.datetime.utcfromtimestamp(0)
                string_micro_seconds = str(
                    (parsed - epoch).total_seconds() * 1000
                ).split(".", 1)[0]
                decimal_value = decimal128.Decimal128(string_micro_seconds)
            else:
                raise TypeError("'%s' type is not supported" % type(parsed))
            return decimal_value

        # Document: https://docs.mongodb.com/manual/reference/operator/aggregation/arrayToObject/
        if operator == "$arrayToObject":
            try:
                parsed = self.parse(values)
            except KeyError:
                return None

            if parsed is None:
                return None

            if not isinstance(parsed, (list, tuple)):
                raise OperationFailure(
                    "$arrayToObject requires an array input, found: {}".format(
                        type(parsed)
                    )
                )

            if all(isinstance(x, dict) and set(x.keys()) == {"k", "v"} for x in parsed):
                return {d["k"]: d["v"] for d in parsed}

            if all(isinstance(x, (list, tuple)) and len(x) == 2 for x in parsed):
                return dict(parsed)

            raise OperationFailure(
                "arrays used with $arrayToObject must contain documents "
                "with k and v fields or two-element arrays"
            )

        # Document: https://docs.mongodb.com/manual/reference/operator/aggregation/objectToArray/
        if operator == "$objectToArray":
            try:
                parsed = self.parse(values)
            except KeyError:
                return None

            if parsed is None:
                return None

            if not isinstance(parsed, (dict, collections.OrderedDict)):
                raise OperationFailure(
                    "$objectToArray requires an object input, found: {}".format(
                        type(parsed)
                    )
                )

            if len(parsed) > 1 and sys.version_info < (3, 6):
                raise NotImplementedError(
                    "Although '%s' is a valid type conversion, it is not implemented for Python 2 "
                    "and Python 3.5 in Mongomock yet." % operator
                )

            return [{"k": k, "v": v} for k, v in parsed.items()]

        raise NotImplementedError(
            "Although '%s' is a valid type conversion operator for the "
            "aggregation pipeline, it is currently not implemented "
            "in Mongomock." % operator
        )

    def _handle_type_operator(self, operator, values):
        # Document: https://docs.mongodb.com/manual/reference/operator/aggregation/isNumber/
        if operator == "$isNumber":
            try:
                parsed = self.parse(values)
            except KeyError:
                return False
            return (
                False
                if isinstance(parsed, bool)
                else isinstance(parsed, numbers.Number)
            )

        # Document: https://docs.mongodb.com/manual/reference/operator/aggregation/isArray/
        if operator == "$isArray":
            try:
                parsed = self.parse(values)
            except KeyError:
                return False
            return isinstance(parsed, (tuple, list))

        raise NotImplementedError(  # pragma: no cover
            "Although '%s' is a valid type operator for the aggregation pipeline, it is currently "
            "not implemented in Mongomock." % operator
        )

    def _handle_conditional_operator(self, operator, values):
        # relative
        from . import SERVER_VERSION

        if operator == "$ifNull":
            fields = values[:-1]
            if len(fields) > 1 and version.parse(SERVER_VERSION) <= version.parse(
                "4.4"
            ):
                raise OperationFailure(
                    "$ifNull supports only one input expression "
                    " in MongoDB v4.4 and lower"
                )
            fallback = values[-1]
            for field in fields:
                try:
                    out_value = self.parse(field)
                    if out_value is not None:
                        return out_value
                except KeyError:
                    pass
            return self.parse(fallback)
        if operator == "$cond":
            if isinstance(values, list):
                condition, true_case, false_case = values
            elif isinstance(values, dict):
                condition = values["if"]
                true_case = values["then"]
                false_case = values["else"]
            condition_value = self._parse_to_bool(condition)
            expression = true_case if condition_value else false_case
            return self.parse(expression)
        # This should never happen: it is only a safe fallback if something went wrong.
        raise NotImplementedError(  # pragma: no cover
            "Although '%s' is a valid conditional operator for the "
            "aggregation pipeline, it is currently not implemented "
            " in Mongomock." % operator
        )

    def _handle_control_flow_operator(self, operator, values):
        if operator == "$switch":
            if not isinstance(values, dict):
                raise OperationFailure(
                    "$switch requires an object as an argument, "
                    "found: %s" % type(values)
                )

            branches = values.get("branches", [])
            if not isinstance(branches, (list, tuple)):
                raise OperationFailure(
                    "$switch expected an array for 'branches', "
                    "found: %s" % type(branches)
                )
            if not branches:
                raise OperationFailure("$switch requires at least one branch.")

            for branch in branches:
                if not isinstance(branch, dict):
                    raise OperationFailure(
                        "$switch expected each branch to be an object, "
                        "found: %s" % type(branch)
                    )
                if "case" not in branch:
                    raise OperationFailure(
                        "$switch requires each branch have a 'case' expression"
                    )
                if "then" not in branch:
                    raise OperationFailure(
                        "$switch requires each branch have a 'then' expression."
                    )

            for branch in branches:
                if self._parse_to_bool(branch["case"]):
                    return self.parse(branch["then"])

            if "default" not in values:
                raise OperationFailure(
                    "$switch could not find a matching branch for an input, "
                    "and no default was specified."
                )
            return self.parse(values["default"])

        # This should never happen: it is only a safe fallback if something went wrong.
        raise NotImplementedError(  # pragma: no cover
            "Although '%s' is a valid control flow operator for the "
            "aggregation pipeline, it is currently not implemented "
            "in Mongomock." % operator
        )

    def _handle_set_operator(self, operator, values):
        if operator == "$in":
            expression, array = values
            return self.parse(expression) in self.parse(array)
        if operator == "$setUnion":
            result = []
            for set_value in values:
                for value in self.parse(set_value):
                    if value not in result:
                        result.append(value)
            return result
        if operator == "$setEquals":
            set_values = [set(self.parse(value)) for value in values]
            for set1, set2 in itertools.combinations(set_values, 2):
                if set1 != set2:
                    return False
            return True
        raise NotImplementedError(
            "Although '%s' is a valid set operator for the aggregation "
            "pipeline, it is currently not implemented in Mongomock." % operator
        )


def _parse_expression(expression, doc_dict, ignore_missing_keys=False):
    """Parse an expression.

    Args:
        expression: an Aggregate Expression, see
            https://docs.mongodb.com/manual/meta/aggregation-quick-reference/#aggregation-expressions.
        doc_dict: the document on which to evaluate the expression.
        ignore_missing_keys: if True, missing keys evaluated by the expression are ignored silently
            if it is possible.
    """
    return _Parser(doc_dict, ignore_missing_keys=ignore_missing_keys).parse(expression)


filtering.register_parse_expression(_parse_expression)


def _accumulate_group(output_fields, group_list):
    doc_dict = {}
    for field, value in output_fields.items():
        if field == "_id":
            continue
        for operator, key in value.items():
            values = []
            for doc in group_list:
                try:
                    values.append(_parse_expression(key, doc))
                except KeyError:
                    continue
            if operator in _GROUPING_OPERATOR_MAP:
                doc_dict[field] = _GROUPING_OPERATOR_MAP[operator](values)
            elif operator == "$addToSet":
                value = []
                val_it = (val or None for val in values)
                # Don't use set in case elt in not hashable (like dicts).
                for elt in val_it:
                    if elt not in value:
                        value.append(elt)
                doc_dict[field] = value
            elif operator == "$push":
                if field not in doc_dict:
                    doc_dict[field] = values
                else:
                    doc_dict[field].extend(values)
            elif operator in group_operators:
                raise NotImplementedError(
                    "Although %s is a valid group operator for the "
                    "aggregation pipeline, it is currently not implemented "
                    "in Mongomock." % operator
                )
            else:
                raise NotImplementedError(
                    "%s is not a valid group operator for the aggregation "
                    "pipeline. See http://docs.mongodb.org/manual/meta/"
                    "aggregation-quick-reference/ for a complete list of "
                    "valid operators." % operator
                )
    return doc_dict


def _fix_sort_key(key_getter):
    def fixed_getter(doc):
        key = key_getter(doc)
        # Convert dictionaries to make sorted() work in Python 3.
        if isinstance(key, dict):
            return [(k, v) for (k, v) in sorted(key.items())]
        return key

    return fixed_getter


def _handle_lookup_stage(in_collection, database, options):
    for operator in ("let", "pipeline"):
        if operator in options:
            raise NotImplementedError(
                "Although '%s' is a valid lookup operator for the "
                "aggregation pipeline, it is currently not "
                "implemented in Mongomock." % operator
            )
    for operator in ("from", "localField", "foreignField", "as"):
        if operator not in options:
            raise OperationFailure("Must specify '%s' field for a $lookup" % operator)
        if not isinstance(options[operator], str):
            raise OperationFailure("Arguments to $lookup must be strings")
        if operator in ("as", "localField", "foreignField") and options[
            operator
        ].startswith("$"):
            raise OperationFailure("FieldPath field names may not start with '$'")
        if operator == "as" and "." in options[operator]:
            raise NotImplementedError(
                "Although '.' is valid in the 'as' "
                "parameters for the lookup stage of the aggregation "
                "pipeline, it is currently not implemented in Mongomock."
            )

    foreign_name = options["from"]
    local_field = options["localField"]
    foreign_field = options["foreignField"]
    local_name = options["as"]
    foreign_collection = database.get_collection(foreign_name)
    for doc in in_collection:
        try:
            query = helpers.get_value_by_dot(doc, local_field)
        except KeyError:
            query = None
        if isinstance(query, list):
            query = {"$in": query}
        matches = foreign_collection.find({foreign_field: query})
        doc[local_name] = [foreign_doc for foreign_doc in matches]

    return in_collection


def _recursive_get(match, nested_fields):
    head = match.get(nested_fields[0])
    remaining_fields = nested_fields[1:]
    if not remaining_fields:
        # Final/last field reached.
        yield head
        return
    # More fields to go, must be list, tuple, or dict.
    if isinstance(head, (list, tuple)):
        for m in head:
            # Yield from _recursive_get(m, remaining_fields).
            for answer in _recursive_get(m, remaining_fields):
                yield answer
    elif isinstance(head, dict):
        # Yield from _recursive_get(head, remaining_fields).
        for answer in _recursive_get(head, remaining_fields):
            yield answer


def _handle_graph_lookup_stage(in_collection, database, options):
    if not isinstance(options.get("maxDepth", 0), int):
        raise OperationFailure("Argument 'maxDepth' to $graphLookup must be a number")
    if not isinstance(options.get("restrictSearchWithMatch", {}), dict):
        raise OperationFailure(
            "Argument 'restrictSearchWithMatch' to $graphLookup must be a Dictionary"
        )
    if not isinstance(options.get("depthField", ""), str):
        raise OperationFailure("Argument 'depthField' to $graphlookup must be a string")
    if "startWith" not in options:
        raise OperationFailure("Must specify 'startWith' field for a $graphLookup")
    for operator in ("as", "connectFromField", "connectToField", "from"):
        if operator not in options:
            raise OperationFailure(
                "Must specify '%s' field for a $graphLookup" % operator
            )
        if not isinstance(options[operator], str):
            raise OperationFailure(
                "Argument '%s' to $graphLookup must be string" % operator
            )
        if options[operator].startswith("$"):
            raise OperationFailure("FieldPath field names may not start with '$'")
        if operator == "as" and "." in options[operator]:
            raise NotImplementedError(
                "Although '.' is valid in the '%s' "
                "parameter for the $graphLookup stage of the aggregation "
                "pipeline, it is currently not implemented in Mongomock." % operator
            )

    foreign_name = options["from"]
    start_with = options["startWith"]
    connect_from_field = options["connectFromField"]
    connect_to_field = options["connectToField"]
    local_name = options["as"]
    max_depth = options.get("maxDepth", None)
    depth_field = options.get("depthField", None)
    restrict_search_with_match = options.get("restrictSearchWithMatch", {})
    foreign_collection = database.get_collection(foreign_name)
    out_doc = copy.deepcopy(in_collection)  # TODO(pascal): speed the deep copy

    def _find_matches_for_depth(query):
        if isinstance(query, list):
            query = {"$in": query}
        matches = foreign_collection.find({connect_to_field: query})
        new_matches = []
        for new_match in matches:
            if (
                filtering.filter_applies(restrict_search_with_match, new_match)
                and new_match["_id"] not in found_items
            ):
                if depth_field is not None:
                    new_match = collections.OrderedDict(
                        new_match, **{depth_field: depth}
                    )
                new_matches.append(new_match)
                found_items.add(new_match["_id"])
        return new_matches

    for doc in out_doc:
        found_items = set()
        depth = 0
        try:
            result = _parse_expression(start_with, doc)
        except KeyError:
            continue
        origin_matches = doc[local_name] = _find_matches_for_depth(result)
        while origin_matches and (max_depth is None or depth < max_depth):
            depth += 1
            newly_discovered_matches = []
            for match in origin_matches:
                nested_fields = connect_from_field.split(".")
                for match_target in _recursive_get(match, nested_fields):
                    newly_discovered_matches += _find_matches_for_depth(match_target)
            doc[local_name] += newly_discovered_matches
            origin_matches = newly_discovered_matches
    return out_doc


def _handle_group_stage(in_collection, unused_database, options):
    grouped_collection = []
    _id = options["_id"]
    if _id:

        def _key_getter(doc):
            try:
                return _parse_expression(_id, doc, ignore_missing_keys=True)
            except KeyError:
                return None

        def _sort_key_getter(doc):
            return filtering.BsonComparable(_key_getter(doc))

        # Sort the collection only for the itertools.groupby.
        # $group does not order its output document.
        sorted_collection = sorted(in_collection, key=_sort_key_getter)
        grouped = itertools.groupby(sorted_collection, _key_getter)
    else:
        grouped = [(None, in_collection)]

    for doc_id, group in grouped:
        group_list = [x for x in group]
        doc_dict = _accumulate_group(options, group_list)
        doc_dict["_id"] = doc_id
        grouped_collection.append(doc_dict)

    return grouped_collection


def _handle_bucket_stage(in_collection, unused_database, options):
    unknown_options = set(options) - {"groupBy", "boundaries", "output", "default"}
    if unknown_options:
        raise OperationFailure(
            "Unrecognized option to $bucket: %s." % unknown_options.pop()
        )
    if "groupBy" not in options or "boundaries" not in options:
        raise OperationFailure(
            "$bucket requires 'groupBy' and 'boundaries' to be specified."
        )
    group_by = options["groupBy"]
    boundaries = options["boundaries"]
    if not isinstance(boundaries, list):
        raise OperationFailure(
            "The $bucket 'boundaries' field must be an array, but found type: %s"
            % type(boundaries)
        )
    if len(boundaries) < 2:
        raise OperationFailure(
            "The $bucket 'boundaries' field must have at least 2 values, but "
            "found %d value(s)." % len(boundaries)
        )
    if sorted(boundaries) != boundaries:
        raise OperationFailure(
            "The 'boundaries' option to $bucket must be sorted in ascending order"
        )
    output_fields = options.get("output", {"count": {"$sum": 1}})
    default_value = options.get("default", None)
    try:
        is_default_last = default_value >= boundaries[-1]
    except TypeError:
        is_default_last = True

    def _get_default_bucket():
        try:
            return options["default"]
        except KeyError as err:
            raise OperationFailure(
                "$bucket could not find a matching branch for "
                "an input, and no default was specified."
            ) from err

    def _get_bucket_id(doc):
        """Get the bucket ID for a document.

        Note that it actually returns a tuple with the first
        param being a sort key to sort the default bucket even
        if it's not the same type as the boundaries.
        """
        try:
            value = _parse_expression(group_by, doc)
        except KeyError:
            return (is_default_last, _get_default_bucket())
        index = bisect.bisect_right(boundaries, value)
        if index and index < len(boundaries):
            return (False, boundaries[index - 1])
        return (is_default_last, _get_default_bucket())

    in_collection = ((_get_bucket_id(doc), doc) for doc in in_collection)
    out_collection = sorted(in_collection, key=lambda kv: kv[0])
    grouped = itertools.groupby(out_collection, lambda kv: kv[0])

    out_collection = []
    for (unused_key, doc_id), group in grouped:
        group_list = [kv[1] for kv in group]
        doc_dict = _accumulate_group(output_fields, group_list)
        doc_dict["_id"] = doc_id
        out_collection.append(doc_dict)
    return out_collection


def _handle_sample_stage(in_collection, unused_database, options):
    if not isinstance(options, dict):
        raise OperationFailure("the $sample stage specification must be an object")
    size = options.pop("size", None)
    if size is None:
        raise OperationFailure("$sample stage must specify a size")
    if options:
        raise OperationFailure(
            "unrecognized option to $sample: %s" % set(options).pop()
        )
    shuffled = list(in_collection)
    _random.shuffle(shuffled)
    return shuffled[:size]


def _handle_sort_stage(in_collection, unused_database, options):
    sort_array = reversed([{x: y} for x, y in options.items()])
    sorted_collection = in_collection
    for sort_pair in sort_array:
        for sortKey, sortDirection in sort_pair.items():
            sorted_collection = sorted(
                sorted_collection,
                key=lambda x: filtering.resolve_sort_key(sortKey, x),
                reverse=sortDirection < 0,
            )
    return sorted_collection


def _handle_unwind_stage(in_collection, unused_database, options):
    if not isinstance(options, dict):
        options = {"path": options}
    path = options["path"]
    if not isinstance(path, str) or path[0] != "$":
        raise ValueError(
            "$unwind failed: exception: field path references must be prefixed "
            "with a '$' '%s'" % path
        )
    path = path[1:]
    should_preserve_null_and_empty = options.get("preserveNullAndEmptyArrays")
    include_array_index = options.get("includeArrayIndex")
    unwound_collection = []
    for doc in in_collection:
        try:
            array_value = helpers.get_value_by_dot(doc, path)
        except KeyError:
            if should_preserve_null_and_empty:
                unwound_collection.append(doc)
            continue
        if array_value is None:
            if should_preserve_null_and_empty:
                unwound_collection.append(doc)
            continue
        if array_value == []:
            if should_preserve_null_and_empty:
                new_doc = copy.deepcopy(doc)
                # We just ran a get_value_by_dot so we know the value exists.
                helpers.delete_value_by_dot(new_doc, path)
                unwound_collection.append(new_doc)
            continue
        if isinstance(array_value, list):
            iter_array = enumerate(array_value)
        else:
            iter_array = [(None, array_value)]
        for index, field_item in iter_array:
            new_doc = copy.deepcopy(doc)
            new_doc = helpers.set_value_by_dot(new_doc, path, field_item)
            if include_array_index:
                new_doc = helpers.set_value_by_dot(new_doc, include_array_index, index)
            unwound_collection.append(new_doc)

    return unwound_collection


# TODO(pascal): Combine with the equivalent function in collection but check
# what are the allowed overriding.
def _combine_projection_spec(filter_list, original_filter, prefix=""):
    """Re-format a projection fields spec into a nested dictionary.

    e.g: ['a', 'b.c', 'b.d'] => {'a': 1, 'b': {'c': 1, 'd': 1}}
    """
    if not isinstance(filter_list, list):
        return filter_list

    filter_dict = collections.OrderedDict()

    for key in filter_list:
        field, separator, subkey = key.partition(".")
        if not separator:
            if isinstance(filter_dict.get(field), list):
                other_key = field + "." + filter_dict[field][0]
                raise OperationFailure(
                    "Invalid $project :: caused by :: specification contains two conflicting paths."
                    " Cannot specify both %s and %s: %s"
                    % (repr(prefix + field), repr(prefix + other_key), original_filter)
                )
            filter_dict[field] = 1
            continue
        if not isinstance(filter_dict.get(field, []), list):
            raise OperationFailure(
                "Invalid $project :: caused by :: specification contains two conflicting paths."
                " Cannot specify both %s and %s: %s"
                % (repr(prefix + field), repr(prefix + key), original_filter)
            )
        filter_dict[field] = filter_dict.get(field, []) + [subkey]

    return collections.OrderedDict(
        (k, _combine_projection_spec(v, original_filter, prefix="%s%s." % (prefix, k)))
        for k, v in filter_dict.items()
    )


def _project_by_spec(doc, proj_spec, is_include):
    output = {}
    for key, value in doc.items():
        if key not in proj_spec:
            if not is_include:
                output[key] = value
            continue

        if not isinstance(proj_spec[key], dict):
            if is_include:
                output[key] = value
            continue

        if isinstance(value, dict):
            output[key] = _project_by_spec(value, proj_spec[key], is_include)
        elif isinstance(value, list):
            output[key] = [
                _project_by_spec(array_value, proj_spec[key], is_include)
                for array_value in value
                if isinstance(array_value, dict)
            ]
        elif not is_include:
            output[key] = value

    return output


def _handle_replace_root_stage(in_collection, unused_database, options):
    if "newRoot" not in options:
        raise OperationFailure(
            "Parameter 'newRoot' is missing for $replaceRoot operation."
        )
    new_root = options["newRoot"]
    out_collection = []
    for doc in in_collection:
        try:
            new_doc = _parse_expression(new_root, doc, ignore_missing_keys=True)
        except KeyError:
            new_doc = None
        if not isinstance(new_doc, dict):
            raise OperationFailure(
                "'newRoot' expression must evaluate to an object, but resulting value was: {}".format(
                    new_doc
                )
            )
        out_collection.append(new_doc)
    return out_collection


def _handle_project_stage(in_collection, unused_database, options):
    filter_list = []
    method = None
    include_id = options.get("_id")
    # Compute new values for each field, except inclusion/exclusions that are
    # handled in one final step.
    new_fields_collection = None
    for field, value in options.items():
        if method is None and (field != "_id" or value):
            method = "include" if value else "exclude"
        elif method == "include" and not value and field != "_id":
            raise OperationFailure(
                "Bad projection specification, cannot exclude fields "
                "other than '_id' in an inclusion projection: %s" % options
            )
        elif method == "exclude" and value:
            raise OperationFailure(
                "Bad projection specification, cannot include fields "
                "or add computed fields during an exclusion projection: %s" % options
            )
        if value in (0, 1, True, False):
            if field != "_id":
                filter_list.append(field)
            continue
        if not new_fields_collection:
            new_fields_collection = [{} for unused_doc in in_collection]

        for in_doc, out_doc in zip(in_collection, new_fields_collection):
            try:
                out_doc[field] = _parse_expression(
                    value, in_doc, ignore_missing_keys=True
                )
            except KeyError:
                # Ignore missing key.
                pass
    if (method == "include") == (include_id is not False and include_id != 0):
        filter_list.append("_id")

    if not filter_list:
        return new_fields_collection

    # Final steps: include or exclude fields and merge with newly created fields.
    projection_spec = _combine_projection_spec(filter_list, original_filter=options)
    out_collection = [
        _project_by_spec(doc, projection_spec, is_include=(method == "include"))
        for doc in in_collection
    ]
    if new_fields_collection:
        return [dict(a, **b) for a, b in zip(out_collection, new_fields_collection)]
    return out_collection


def _handle_add_fields_stage(in_collection, unused_database, options):
    if not options:
        raise OperationFailure(
            "Invalid $addFields :: caused by :: specification must have at least one field"
        )
    out_collection = [dict(doc) for doc in in_collection]
    for field, value in options.items():
        for in_doc, out_doc in zip(in_collection, out_collection):
            try:
                out_value = _parse_expression(value, in_doc, ignore_missing_keys=True)
            except KeyError:
                continue
            parts = field.split(".")
            for subfield in parts[:-1]:
                out_doc[subfield] = out_doc.get(subfield, {})
                if not isinstance(out_doc[subfield], dict):
                    out_doc[subfield] = {}
                out_doc = out_doc[subfield]
            out_doc[parts[-1]] = out_value
    return out_collection


def _handle_out_stage(in_collection, database, options):
    # TODO(MetrodataTeam): should leave the origin collection unchanged
    out_collection = database.get_collection(options)
    if out_collection.find_one():
        out_collection.drop()
    if in_collection:
        out_collection.insert_many(in_collection)
    return in_collection


def _handle_count_stage(in_collection, database, options):
    if not isinstance(options, str) or options == "":
        raise OperationFailure("the count field must be a non-empty string")
    elif options.startswith("$"):
        raise OperationFailure("the count field cannot be a $-prefixed path")
    elif "." in options:
        raise OperationFailure("the count field cannot contain '.'")
    return [{options: len(in_collection)}]


def _handle_facet_stage(in_collection, database, options):
    out_collection_by_pipeline = {}
    for pipeline_title, pipeline in options.items():
        out_collection_by_pipeline[pipeline_title] = list(
            process_pipeline(in_collection, database, pipeline, None)
        )
    return [out_collection_by_pipeline]


def _handle_match_stage(in_collection, database, options):
    spec = helpers.patch_datetime_awareness_in_document(options)
    return [
        doc
        for doc in in_collection
        if filtering.filter_applies(
            spec, helpers.patch_datetime_awareness_in_document(doc)
        )
    ]


_PIPELINE_HANDLERS = {
    "$addFields": _handle_add_fields_stage,
    "$bucket": _handle_bucket_stage,
    "$bucketAuto": None,
    "$collStats": None,
    "$count": _handle_count_stage,
    "$currentOp": None,
    "$facet": _handle_facet_stage,
    "$geoNear": None,
    "$graphLookup": _handle_graph_lookup_stage,
    "$group": _handle_group_stage,
    "$indexStats": None,
    "$limit": lambda c, d, o: c[:o],
    "$listLocalSessions": None,
    "$listSessions": None,
    "$lookup": _handle_lookup_stage,
    "$match": _handle_match_stage,
    "$merge": None,
    "$out": _handle_out_stage,
    "$planCacheStats": None,
    "$project": _handle_project_stage,
    "$redact": None,
    "$replaceRoot": _handle_replace_root_stage,
    "$replaceWith": None,
    "$sample": _handle_sample_stage,
    "$set": _handle_add_fields_stage,
    "$skip": lambda c, d, o: c[o:],
    "$sort": _handle_sort_stage,
    "$sortByCount": None,
    "$unset": None,
    "$unwind": _handle_unwind_stage,
}


def process_pipeline(collection, database, pipeline, session):
    if session:
        raise NotImplementedError("Mongomock does not handle sessions yet")

    for stage in pipeline:
        for operator, options in stage.items():
            try:
                handler = _PIPELINE_HANDLERS[operator]
            except KeyError as err:
                raise NotImplementedError(
                    "%s is not a valid operator for the aggregation pipeline. "
                    "See http://docs.mongodb.org/manual/meta/aggregation-quick-reference/ "
                    "for a complete list of valid operators." % operator
                ) from err
            if not handler:
                raise NotImplementedError(
                    "Although '%s' is a valid operator for the aggregation pipeline, it is "
                    "currently not implemented in Mongomock." % operator
                )
            collection = handler(collection, database, options)

    return command_cursor.CommandCursor(collection)
