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

import pytz

# third party
from packaging import version

# relative
from . import OperationFailure, command_cursor, filtering, helpers

try:
    # third party
    from bson import Regex, decimal128
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
comparison_operators = ["$cmp", "$eq", "$ne", *list(filtering.SORTING_OPERATOR_MAP.keys())]
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
    values_list = [v for v in values if isinstance(v, numbers.Number)]
    if not values_list:
        return None
    return sum(values_list) / float(len(list(values_list)))


def _group_operation(values, operator):
    values_list = [v for v in values if v is not None]
    if not values_list:
        return None
    return operator(values_list)


def _sum_operation(values):
    values_list = []
    if decimal_support:
        for v in values:
            if isinstance(v, numbers.Number):
                values_list.append(v)
            elif isinstance(v, decimal128.Decimal128):
                values_list.append(v.to_decimal())
    else:
        values_list = [v for v in values if isinstance(v, numbers.Number)]
    sum_value = sum(values_list)
    return (
        decimal128.Decimal128(sum_value)
        if isinstance(sum_value, decimal.Decimal)
        else sum_value
    )


def _merge_objects_operation(values):
    merged_doc = {}
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


class _Parser:
    """Helper to parse expressions within the aggregate pipeline."""

    def __init__(self, doc_dict, user_vars=None, ignore_missing_keys=False) -> None:
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
                % (len(expression), expression),
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
                msg = f"'{k}' is a valid operation but it is not supported by Mongomock yet."
                raise NotImplementedError(
                    msg,
                )
            if k.startswith("$"):
                msg = f"Unrecognized expression '{k}'"
                raise OperationFailure(msg)
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
        """Parse a MongoDB expression and then convert it to bool."""
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
                self._doc_dict, expression[1:], can_generate_array=True,
            )
        return expression

    def _handle_boolean_operator(self, operator, values):
        if operator == "$and":
            return all(self._parse_to_bool(value) for value in values)
        if operator == "$or":
            return any(self._parse_to_bool(value) for value in values)
        if operator == "$not":
            return not self._parse_to_bool(values)
        # This should never happen: it is only a safe fallback if something went wrong.
        msg = (
            f"Although '{operator}' is a valid boolean operator for the "
            "aggregation pipeline, it is currently not implemented"
            " in Mongomock."
        )
        raise NotImplementedError(  # pragma: no cover
            msg,
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
                msg = f"Parameter to {operator} must evaluate to a number, got '{type(number)}'"
                raise OperationFailure(
                    msg,
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
            if not isinstance(values, tuple | list):
                msg = f"Parameter to {operator} must evaluate to a list, got '{type(values)}'"
                raise OperationFailure(
                    msg,
                )

            if len(values) != 2:
                msg = f"{operator} must have only 2 parameters"
                raise OperationFailure(msg)
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
                    number_1, int | float,
                ):
                    number_1 = datetime.timedelta(milliseconds=number_1)
                res = number_0 - number_1
                if isinstance(res, datetime.timedelta):
                    return round(res.total_seconds() * 1000)
                return res

        assert isinstance(values, tuple | list), (
            f"Parameter to {operator} must evaluate to a list, got '{type(values)}'"
        )

        parsed_values = list(self.parse_many(values))
        assert parsed_values, f"{operator} must have at least one parameter"
        for value in parsed_values:
            if value is None:
                return None
            assert isinstance(value, numbers.Number), f"{operator} only uses numbers"
        if operator == "$add":
            return sum(parsed_values)
        if operator == "$multiply":
            return functools.reduce(lambda x, y: x * y, parsed_values)

        # This should never happen: it is only a safe fallback if something went wrong.
        msg = (
            f"Although '{operator}' is a valid aritmetic operator for the aggregation "
            "pipeline, it is currently not implemented  in Mongomock."
        )
        raise NotImplementedError(  # pragma: no cover
            msg,
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
                msg = "Array have length less than index value"
                raise KeyError(msg) from error

        msg = (
            f"Although '{operator}' is a valid project operator for the "
            "aggregation pipeline, it is currently not implemented "
            "in Mongomock."
        )
        raise NotImplementedError(
            msg,
        )

    def _handle_projection_operator(self, operator, value):
        if operator == "$literal":
            return value
        if operator == "$let":
            if not isinstance(value, dict):
                msg = "$let only supports an object as its argument"
                raise InvalidDocument(msg)
            for field in ("vars", "in"):
                if field not in value:
                    msg = f"Missing '{field}' parameter to $let"
                    raise OperationFailure(
                        msg,
                    )
            if not isinstance(value["vars"], dict):
                msg = "invalid parameter: expected an object (vars)"
                raise OperationFailure(msg)
            user_vars = {
                var_key: self.parse(var_value)
                for var_key, var_value in value["vars"].items()
            }
            return _Parser(
                self._doc_dict,
                dict(self._user_vars, **user_vars),
                ignore_missing_keys=self._ignore_missing_keys,
            ).parse(value["in"])
        msg = (
            f"Although '{operator}' is a valid project operator for the "
            "aggregation pipeline, it is currently not implemented "
            "in Mongomock."
        )
        raise NotImplementedError(
            msg,
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
                filtering.SORTING_OPERATOR_MAP[operator], a, b,
            )
        msg = (
            f"Although '{operator}' is a valid comparison operator for the "
            "aggregation pipeline, it is currently not implemented "
            " in Mongomock."
        )
        raise NotImplementedError(
            msg,
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
                msg = "split must have 2 items"
                raise OperationFailure(msg)
            try:
                string = self.parse(values[0])
                delimiter = self.parse(values[1])
            except KeyError:
                return None

            if string is None or delimiter is None:
                return None
            if not isinstance(string, str):
                msg = "split first argument must evaluate to string"
                raise TypeError(msg)
            if not isinstance(delimiter, str):
                msg = "split second argument must evaluate to string"
                raise TypeError(msg)
            return string.split(delimiter)
        if operator == "$substr":
            if len(values) != 3:
                msg = "substr must have 3 items"
                raise OperationFailure(msg)
            string = str(self.parse(values[0]))
            first = self.parse(values[1])
            length = self.parse(values[2])
            if string is None:
                return ""
            if first < 0:
                warnings.warn(
                    "Negative starting point given to $substr is accepted only until "
                    "MongoDB 3.7. This behavior will change in the future.",
                )
                return ""
            if length < 0:
                warnings.warn(
                    "Negative length given to $substr is accepted only until "
                    "MongoDB 3.7. This behavior will change in the future.",
                )
            second = len(string) if length < 0 else first + length
            return string[first:second]
        if operator == "$strcasecmp":
            if len(values) != 2:
                msg = "strcasecmp must have 2 items"
                raise OperationFailure(msg)
            a, b = str(self.parse(values[0])), str(self.parse(values[1]))
            return 0 if a == b else -1 if a < b else 1
        if operator == "$regexMatch":
            if not isinstance(values, dict):
                msg = f"$regexMatch expects an object of named arguments but found: {type(values)}"
                raise OperationFailure(
                    msg,
                )
            for field in ("input", "regex"):
                if field not in values:
                    msg = f"$regexMatch requires '{field}' parameter"
                    raise OperationFailure(
                        msg,
                    )
            unknown_args = set(values) - {"input", "regex", "options"}
            if unknown_args:
                msg = f"$regexMatch found an unknown argument: {next(iter(unknown_args))}"
                raise OperationFailure(
                    msg,
                )

            try:
                input_value = self.parse(values["input"])
            except KeyError:
                return False
            if not isinstance(input_value, str):
                msg = "$regexMatch needs 'input' to be of type string"
                raise OperationFailure(msg)

            try:
                regex_val = self.parse(values["regex"])
            except KeyError:
                return False
            options = None
            for option in values.get("options", ""):
                if option not in "imxs":
                    msg = f"$regexMatch invalid flag in regex options: {option}"
                    raise OperationFailure(
                        msg,
                    )
                re_option = getattr(re, option.upper())
                if options is None:
                    options = re_option
                else:
                    options |= re_option
            if isinstance(regex_val, str):
                regex = re.compile(regex_val) if options is None else re.compile(regex_val, options)
            elif "options" in values and regex_val.flags:
                msg = "$regexMatch: regex option(s) specified in both 'regex' and 'option' fields"
                raise OperationFailure(
                    msg,
                )
            elif isinstance(regex_val, helpers.RE_TYPE):
                if options and not regex_val.flags:
                    regex = re.compile(regex_val.pattern, options)
                elif regex_val.flags & ~(re.IGNORECASE | re.MULTILINE | re.VERBOSE | re.DOTALL):
                    msg = f"$regexMatch invalid flag in regex options: {regex_val.flags}"
                    raise OperationFailure(
                        msg,
                    )
                else:
                    regex = regex_val
            elif isinstance(regex_val, _RE_TYPES):
                # bson.Regex
                if regex_val.flags & ~(re.IGNORECASE | re.MULTILINE | re.VERBOSE | re.DOTALL):
                    msg = f"$regexMatch invalid flag in regex options: {regex_val.flags}"
                    raise OperationFailure(
                        msg,
                    )
                regex = re.compile(regex_val.pattern, regex_val.flags or options)
            else:
                msg = "$regexMatch needs 'regex' to be of type string or regex"
                raise OperationFailure(
                    msg,
                )

            return bool(regex.search(input_value))

        # This should never happen: it is only a safe fallback if something went wrong.
        msg = (
            f"Although '{operator}' is a valid string operator for the aggregation "
            "pipeline, it is currently not implemented  in Mongomock."
        )
        raise NotImplementedError(  # pragma: no cover
            msg,
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
                msg = (
                    "$dateToString operator must correspond a dict"
                    'that has "format" and "date" field.'
                )
                raise OperationFailure(
                    msg,
                )
            if not isinstance(values, dict) or not {"format", "date"} <= set(values):
                msg = (
                    "$dateToString operator must correspond a dict"
                    'that has "format" and "date" field.'
                )
                raise OperationFailure(
                    msg,
                )
            if "%L" in out_value["format"]:
                msg = (
                    "Although %L is a valid date format for the "
                    "$dateToString operator, it is currently not implemented "
                    " in Mongomock."
                )
                raise NotImplementedError(
                    msg,
                )
            if "onNull" in values:
                msg = (
                    "Although onNull is a valid field for the "
                    "$dateToString operator, it is currently not implemented "
                    " in Mongomock."
                )
                raise NotImplementedError(
                    msg,
                )
            if "timezone" in values:
                msg = (
                    "Although timezone is a valid field for the "
                    "$dateToString operator, it is currently not implemented "
                    " in Mongomock."
                )
                raise NotImplementedError(
                    msg,
                )
            return out_value["date"].strftime(out_value["format"])
        if operator == "$dateFromParts":
            if not isinstance(out_value, dict):
                msg = (
                    f"{operator} operator must correspond a dict "
                    'that has "year" or "isoWeekYear" field.'
                )
                raise OperationFailure(
                    msg,
                )
            if len(set(out_value) & {"year", "isoWeekYear"}) != 1:
                msg = (
                    f"{operator} operator must correspond a dict "
                    'that has "year" or "isoWeekYear" field.'
                )
                raise OperationFailure(
                    msg,
                )
            for field in ("isoWeekYear", "isoWeek", "isoDayOfWeek", "timezone"):
                if field in out_value:
                    msg = (
                        f"Although {field} is a valid field for the "
                        f"{operator} operator, it is currently not implemented "
                        "in Mongomock."
                    )
                    raise NotImplementedError(
                        msg,
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

        msg = (
            f"Although '{operator}' is a valid date operator for the "
            "aggregation pipeline, it is currently not implemented "
            " in Mongomock."
        )
        raise NotImplementedError(
            msg,
        )

    def _handle_array_operator(self, operator, value):
        if operator == "$concatArrays":
            if not isinstance(value, list | tuple):
                value = [value]

            parsed_list = list(self.parse_many(value))
            for parsed_item in parsed_list:
                if parsed_item is not None and not isinstance(
                    parsed_item, list | tuple,
                ):
                    msg = (
                        f"$concatArrays only supports arrays, not {type(parsed_item)}"
                    )
                    raise OperationFailure(
                        msg,
                    )

            return (
                None
                if None in parsed_list
                else list(itertools.chain.from_iterable(parsed_list))
            )

        if operator == "$map":
            if not isinstance(value, dict):
                msg = "$map only supports an object as its argument"
                raise OperationFailure(msg)

            # NOTE: while the two validations below could be achieved with
            # one-liner set operations (e.g. set(value) - {'input', 'as',
            # 'in'}), we prefer the iteration-based approaches in order to
            # mimic MongoDB's behavior regarding the order of evaluation. For
            # example, MongoDB complains about 'input' parameter missing before
            # 'in'.
            for k in ("input", "in"):
                if k not in value:
                    msg = f"Missing '{k}' parameter to $map"
                    raise OperationFailure(msg)

            for k in value:
                if k not in {"input", "as", "in"}:
                    msg = f"Unrecognized parameter to $map: {k}"
                    raise OperationFailure(msg)

            input_array = self._parse_or_None(value["input"])

            if input_array is None or input_array is None:
                return None

            if not isinstance(input_array, list | tuple):
                msg = f"input to $map must be an array not {type(input_array)}"
                raise OperationFailure(
                    msg,
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
                        "%d were passed in." % len(value),
                    )
                value = value[0]
            array_value = self._parse_or_None(value)
            if not isinstance(array_value, list | tuple):
                raise OperationFailure(
                    "The argument to $size must be an array, but was of type: %s"
                    % ("missing" if array_value is None else type(array_value)),
                )
            return len(array_value)

        if operator == "$filter":
            if not isinstance(value, dict):
                msg = "$filter only supports an object as its argument"
                raise OperationFailure(
                    msg,
                )
            extra_params = set(value) - {"input", "cond", "as"}
            if extra_params:
                msg = f"Unrecognized parameter to $filter: {extra_params.pop()}"
                raise OperationFailure(
                    msg,
                )
            missing_params = {"input", "cond"} - set(value)
            if missing_params:
                msg = f"Missing '{missing_params.pop()}' parameter to $filter"
                raise OperationFailure(
                    msg,
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
                msg = "$slice only supports a list as its argument"
                raise OperationFailure(msg)
            if len(value) < 2 or len(value) > 3:
                msg = (
                    "Expression $slice takes at least 2 arguments, and at most "
                    f"3, but {len(value)} were passed in"
                )
                raise OperationFailure(
                    msg,
                )
            array_value = self.parse(value[0])
            if not isinstance(array_value, list):
                msg = (
                    f"First argument to $slice must be an array, but is of type: {type(array_value)}"
                )
                raise OperationFailure(
                    msg,
                )
            for num, v in zip(("Second", "Third"), value[1:]):
                if not isinstance(v, int):
                    msg = (
                        f"{num} argument to $slice must be numeric, but is of type: {type(v)}"
                    )
                    raise OperationFailure(
                        msg,
                    )
            if len(value) > 2 and value[2] <= 0:
                msg = f"Third argument to $slice must be positive: {value[2]}"
                raise OperationFailure(
                    msg,
                )

            start = value[1]
            if start < 0:
                stop = len(array_value) + start + value[2] if len(value) > 2 else None
            elif len(value) > 2:
                stop = start + value[2]
            else:
                stop = start
                start = 0
            return array_value[start:stop]

        msg = (
            f"Although '{operator}' is a valid array operator for the "
            "aggregation pipeline, it is currently not implemented "
            "in Mongomock."
        )
        raise NotImplementedError(
            msg,
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
            msg = "You need to import the pymongo library to support decimal128 type."
            raise NotImplementedError(
                msg,
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
            msg = "You need to import the pymongo library to support decimal128 type."
            raise NotImplementedError(
                msg,
            )

        # Document: https://docs.mongodb.com/manual/reference/operator/aggregation/toDecimal/
        if operator == "$toDecimal":
            if not decimal_support:
                msg = "You need to import the pymongo library to support decimal128 type."
                raise NotImplementedError(
                    msg,
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
                    msg = (
                        f"Failed to parse number '{parsed}' in $convert with no onError value:"
                        "Failed to parse string to decimal"
                    )
                    raise OperationFailure(
                        msg,
                    ) from err
            elif isinstance(parsed, datetime.datetime):
                epoch = datetime.datetime.utcfromtimestamp(0)
                string_micro_seconds = str(
                    (parsed - epoch).total_seconds() * 1000,
                ).split(".", 1)[0]
                decimal_value = decimal128.Decimal128(string_micro_seconds)
            else:
                msg = f"'{type(parsed)}' type is not supported"
                raise TypeError(msg)
            return decimal_value

        # Document: https://docs.mongodb.com/manual/reference/operator/aggregation/arrayToObject/
        if operator == "$arrayToObject":
            try:
                parsed = self.parse(values)
            except KeyError:
                return None

            if parsed is None:
                return None

            if not isinstance(parsed, list | tuple):
                msg = (
                    f"$arrayToObject requires an array input, found: {type(parsed)}"
                )
                raise OperationFailure(
                    msg,
                )

            if all(isinstance(x, dict) and set(x.keys()) == {"k", "v"} for x in parsed):
                return {d["k"]: d["v"] for d in parsed}

            if all(isinstance(x, list | tuple) and len(x) == 2 for x in parsed):
                return dict(parsed)

            msg = (
                "arrays used with $arrayToObject must contain documents "
                "with k and v fields or two-element arrays"
            )
            raise OperationFailure(
                msg,
            )

        # Document: https://docs.mongodb.com/manual/reference/operator/aggregation/objectToArray/
        if operator == "$objectToArray":
            try:
                parsed = self.parse(values)
            except KeyError:
                return None

            if parsed is None:
                return None

            if not isinstance(parsed, dict | collections.OrderedDict):
                msg = (
                    f"$objectToArray requires an object input, found: {type(parsed)}"
                )
                raise OperationFailure(
                    msg,
                )

            if len(parsed) > 1 and sys.version_info < (3, 6):
                msg = (
                    f"Although '{operator}' is a valid type conversion, it is not implemented for Python 2 "
                    "and Python 3.5 in Mongomock yet."
                )
                raise NotImplementedError(
                    msg,
                )

            return [{"k": k, "v": v} for k, v in parsed.items()]

        msg = (
            f"Although '{operator}' is a valid type conversion operator for the "
            "aggregation pipeline, it is currently not implemented "
            "in Mongomock."
        )
        raise NotImplementedError(
            msg,
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
            return isinstance(parsed, tuple | list)

        msg = (
            f"Although '{operator}' is a valid type operator for the aggregation pipeline, it is currently "
            "not implemented in Mongomock."
        )
        raise NotImplementedError(  # pragma: no cover
            msg,
        )

    def _handle_conditional_operator(self, operator, values):
        # relative
        from . import SERVER_VERSION

        if operator == "$ifNull":
            fields = values[:-1]
            if len(fields) > 1 and version.parse(SERVER_VERSION) <= version.parse(
                "4.4",
            ):
                msg = (
                    "$ifNull supports only one input expression "
                    " in MongoDB v4.4 and lower"
                )
                raise OperationFailure(
                    msg,
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
        msg = (
            f"Although '{operator}' is a valid conditional operator for the "
            "aggregation pipeline, it is currently not implemented "
            " in Mongomock."
        )
        raise NotImplementedError(  # pragma: no cover
            msg,
        )

    def _handle_control_flow_operator(self, operator, values):
        if operator == "$switch":
            if not isinstance(values, dict):
                msg = (
                    "$switch requires an object as an argument, "
                    f"found: {type(values)}"
                )
                raise OperationFailure(
                    msg,
                )

            branches = values.get("branches", [])
            if not isinstance(branches, list | tuple):
                msg = (
                    "$switch expected an array for 'branches', "
                    f"found: {type(branches)}"
                )
                raise OperationFailure(
                    msg,
                )
            if not branches:
                msg = "$switch requires at least one branch."
                raise OperationFailure(msg)

            for branch in branches:
                if not isinstance(branch, dict):
                    msg = (
                        "$switch expected each branch to be an object, "
                        f"found: {type(branch)}"
                    )
                    raise OperationFailure(
                        msg,
                    )
                if "case" not in branch:
                    msg = "$switch requires each branch have a 'case' expression"
                    raise OperationFailure(
                        msg,
                    )
                if "then" not in branch:
                    msg = "$switch requires each branch have a 'then' expression."
                    raise OperationFailure(
                        msg,
                    )

            for branch in branches:
                if self._parse_to_bool(branch["case"]):
                    return self.parse(branch["then"])

            if "default" not in values:
                msg = (
                    "$switch could not find a matching branch for an input, "
                    "and no default was specified."
                )
                raise OperationFailure(
                    msg,
                )
            return self.parse(values["default"])

        # This should never happen: it is only a safe fallback if something went wrong.
        msg = (
            f"Although '{operator}' is a valid control flow operator for the "
            "aggregation pipeline, it is currently not implemented "
            "in Mongomock."
        )
        raise NotImplementedError(  # pragma: no cover
            msg,
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
            return all(set1 == set2 for set1, set2 in itertools.combinations(set_values, 2))
        msg = (
            f"Although '{operator}' is a valid set operator for the aggregation "
            "pipeline, it is currently not implemented in Mongomock."
        )
        raise NotImplementedError(
            msg,
        )


def _parse_expression(expression, doc_dict, ignore_missing_keys=False):
    """Parse an expression.

    Args:
    ----
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
                msg = (
                    f"Although {operator} is a valid group operator for the "
                    "aggregation pipeline, it is currently not implemented "
                    "in Mongomock."
                )
                raise NotImplementedError(
                    msg,
                )
            else:
                msg = (
                    f"{operator} is not a valid group operator for the aggregation "
                    "pipeline. See http://docs.mongodb.org/manual/meta/"
                    "aggregation-quick-reference/ for a complete list of "
                    "valid operators."
                )
                raise NotImplementedError(
                    msg,
                )
    return doc_dict


def _fix_sort_key(key_getter):
    def fixed_getter(doc):
        key = key_getter(doc)
        # Convert dictionaries to make sorted() work in Python 3.
        if isinstance(key, dict):
            return sorted(key.items())
        return key

    return fixed_getter


def _handle_lookup_stage(in_collection, database, options):
    for operator in ("let", "pipeline"):
        if operator in options:
            msg = (
                f"Although '{operator}' is a valid lookup operator for the "
                "aggregation pipeline, it is currently not "
                "implemented in Mongomock."
            )
            raise NotImplementedError(
                msg,
            )
    for operator in ("from", "localField", "foreignField", "as"):
        if operator not in options:
            msg = f"Must specify '{operator}' field for a $lookup"
            raise OperationFailure(msg)
        if not isinstance(options[operator], str):
            msg = "Arguments to $lookup must be strings"
            raise OperationFailure(msg)
        if operator in ("as", "localField", "foreignField") and options[
            operator
        ].startswith("$"):
            msg = "FieldPath field names may not start with '$'"
            raise OperationFailure(msg)
        if operator == "as" and "." in options[operator]:
            msg = (
                "Although '.' is valid in the 'as' "
                "parameters for the lookup stage of the aggregation "
                "pipeline, it is currently not implemented in Mongomock."
            )
            raise NotImplementedError(
                msg,
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
        doc[local_name] = list(matches)

    return in_collection


def _recursive_get(match, nested_fields):
    head = match.get(nested_fields[0])
    remaining_fields = nested_fields[1:]
    if not remaining_fields:
        # Final/last field reached.
        yield head
        return
    # More fields to go, must be list, tuple, or dict.
    if isinstance(head, list | tuple):
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
        msg = "Argument 'maxDepth' to $graphLookup must be a number"
        raise OperationFailure(msg)
    if not isinstance(options.get("restrictSearchWithMatch", {}), dict):
        msg = "Argument 'restrictSearchWithMatch' to $graphLookup must be a Dictionary"
        raise OperationFailure(
            msg,
        )
    if not isinstance(options.get("depthField", ""), str):
        msg = "Argument 'depthField' to $graphlookup must be a string"
        raise OperationFailure(msg)
    if "startWith" not in options:
        msg = "Must specify 'startWith' field for a $graphLookup"
        raise OperationFailure(msg)
    for operator in ("as", "connectFromField", "connectToField", "from"):
        if operator not in options:
            msg = f"Must specify '{operator}' field for a $graphLookup"
            raise OperationFailure(
                msg,
            )
        if not isinstance(options[operator], str):
            msg = f"Argument '{operator}' to $graphLookup must be string"
            raise OperationFailure(
                msg,
            )
        if options[operator].startswith("$"):
            msg = "FieldPath field names may not start with '$'"
            raise OperationFailure(msg)
        if operator == "as" and "." in options[operator]:
            msg = (
                f"Although '.' is valid in the '{operator}' "
                "parameter for the $graphLookup stage of the aggregation "
                "pipeline, it is currently not implemented in Mongomock."
            )
            raise NotImplementedError(
                msg,
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
                        new_match, **{depth_field: depth},
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
        group_list = list(group)
        doc_dict = _accumulate_group(options, group_list)
        doc_dict["_id"] = doc_id
        grouped_collection.append(doc_dict)

    return grouped_collection


def _handle_bucket_stage(in_collection, unused_database, options):
    unknown_options = set(options) - {"groupBy", "boundaries", "output", "default"}
    if unknown_options:
        msg = f"Unrecognized option to $bucket: {unknown_options.pop()}."
        raise OperationFailure(
            msg,
        )
    if "groupBy" not in options or "boundaries" not in options:
        msg = "$bucket requires 'groupBy' and 'boundaries' to be specified."
        raise OperationFailure(
            msg,
        )
    group_by = options["groupBy"]
    boundaries = options["boundaries"]
    if not isinstance(boundaries, list):
        msg = f"The $bucket 'boundaries' field must be an array, but found type: {type(boundaries)}"
        raise OperationFailure(
            msg,
        )
    if len(boundaries) < 2:
        raise OperationFailure(
            "The $bucket 'boundaries' field must have at least 2 values, but "
            "found %d value(s)." % len(boundaries),
        )
    if sorted(boundaries) != boundaries:
        msg = "The 'boundaries' option to $bucket must be sorted in ascending order"
        raise OperationFailure(
            msg,
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
            msg = (
                "$bucket could not find a matching branch for "
                "an input, and no default was specified."
            )
            raise OperationFailure(
                msg,
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
    for (_unused_key, doc_id), group in grouped:
        group_list = [kv[1] for kv in group]
        doc_dict = _accumulate_group(output_fields, group_list)
        doc_dict["_id"] = doc_id
        out_collection.append(doc_dict)
    return out_collection


def _handle_sample_stage(in_collection, unused_database, options):
    if not isinstance(options, dict):
        msg = "the $sample stage specification must be an object"
        raise OperationFailure(msg)
    size = options.pop("size", None)
    if size is None:
        msg = "$sample stage must specify a size"
        raise OperationFailure(msg)
    if options:
        msg = f"unrecognized option to $sample: {set(options).pop()}"
        raise OperationFailure(
            msg,
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
        msg = (
            "$unwind failed: exception: field path references must be prefixed "
            f"with a '$' '{path}'"
        )
        raise ValueError(
            msg,
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
        iter_array = enumerate(array_value) if isinstance(array_value, list) else [(None, array_value)]
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
                msg = (
                    "Invalid $project :: caused by :: specification contains two conflicting paths."
                    f" Cannot specify both {prefix + field!r} and {prefix + other_key!r}: {original_filter}"
                )
                raise OperationFailure(
                    msg,
                )
            filter_dict[field] = 1
            continue
        if not isinstance(filter_dict.get(field, []), list):
            msg = (
                "Invalid $project :: caused by :: specification contains two conflicting paths."
                f" Cannot specify both {prefix + field!r} and {prefix + key!r}: {original_filter}"
            )
            raise OperationFailure(
                msg,
            )
        filter_dict[field] = [*filter_dict.get(field, []), subkey]

    return collections.OrderedDict(
        (k, _combine_projection_spec(v, original_filter, prefix=f"{prefix}{k}."))
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
        msg = "Parameter 'newRoot' is missing for $replaceRoot operation."
        raise OperationFailure(
            msg,
        )
    new_root = options["newRoot"]
    out_collection = []
    for doc in in_collection:
        try:
            new_doc = _parse_expression(new_root, doc, ignore_missing_keys=True)
        except KeyError:
            new_doc = None
        if not isinstance(new_doc, dict):
            msg = (
                f"'newRoot' expression must evaluate to an object, but resulting value was: {new_doc}"
            )
            raise OperationFailure(
                msg,
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
            msg = (
                "Bad projection specification, cannot exclude fields "
                f"other than '_id' in an inclusion projection: {options}"
            )
            raise OperationFailure(
                msg,
            )
        elif method == "exclude" and value:
            msg = (
                "Bad projection specification, cannot include fields "
                f"or add computed fields during an exclusion projection: {options}"
            )
            raise OperationFailure(
                msg,
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
                    value, in_doc, ignore_missing_keys=True,
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
        msg = "Invalid $addFields :: caused by :: specification must have at least one field"
        raise OperationFailure(
            msg,
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
        msg = "the count field must be a non-empty string"
        raise OperationFailure(msg)
    elif options.startswith("$"):
        msg = "the count field cannot be a $-prefixed path"
        raise OperationFailure(msg)
    elif "." in options:
        msg = "the count field cannot contain '.'"
        raise OperationFailure(msg)
    return [{options: len(in_collection)}]


def _handle_facet_stage(in_collection, database, options):
    out_collection_by_pipeline = {}
    for pipeline_title, pipeline in options.items():
        out_collection_by_pipeline[pipeline_title] = list(
            process_pipeline(in_collection, database, pipeline, None),
        )
    return [out_collection_by_pipeline]


def _handle_match_stage(in_collection, database, options):
    spec = helpers.patch_datetime_awareness_in_document(options)
    return [
        doc
        for doc in in_collection
        if filtering.filter_applies(
            spec, helpers.patch_datetime_awareness_in_document(doc),
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
        msg = "Mongomock does not handle sessions yet"
        raise NotImplementedError(msg)

    for stage in pipeline:
        for operator, options in stage.items():
            try:
                handler = _PIPELINE_HANDLERS[operator]
            except KeyError as err:
                msg = (
                    f"{operator} is not a valid operator for the aggregation pipeline. "
                    "See http://docs.mongodb.org/manual/meta/aggregation-quick-reference/ "
                    "for a complete list of valid operators."
                )
                raise NotImplementedError(
                    msg,
                ) from err
            if not handler:
                msg = (
                    f"Although '{operator}' is a valid operator for the aggregation pipeline, it is "
                    "currently not implemented in Mongomock."
                )
                raise NotImplementedError(
                    msg,
                )
            collection = handler(collection, database, options)

    return command_cursor.CommandCursor(collection)
