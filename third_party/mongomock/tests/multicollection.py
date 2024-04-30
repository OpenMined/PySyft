# stdlib
from collections import OrderedDict
import copy
import functools

# third party
from mongomock.helpers import RE_TYPE

# relative
from .diff import diff

_COMPARE_EXCEPTIONS = "exceptions"


class MultiCollection(object):
    def __init__(self, conns):
        super(MultiCollection, self).__init__()
        self.conns = conns.copy()
        self.do = Foreach(self.conns, compare=False)
        self.compare = Foreach(self.conns, compare=True)
        self.compare_ignore_order = Foreach(self.conns, compare=True, ignore_order=True)
        self.compare_exceptions = Foreach(self.conns, compare=_COMPARE_EXCEPTIONS)


class Foreach(object):
    def __init__(self, objs, compare, ignore_order=False, method_result_decorators=()):
        self.___objs = objs
        self.___compare = compare
        self.___ignore_order = ignore_order
        self.___decorators = list(method_result_decorators)
        self.___sort_by = None

    def __getattr__(self, method_name):
        return ForeachMethod(
            self.___objs,
            self.___compare,
            self.___ignore_order,
            method_name,
            self.___decorators,
            self.___sort_by,
        )

    def sort_by(self, fun):
        self.___sort_by = fun
        return self

    def __call__(self, *decorators):
        return Foreach(
            self.___objs,
            self.___compare,
            self.___ignore_order,
            self.___decorators + list(decorators),
        )


class ForeachMethod(object):
    def __init__(self, objs, compare, ignore_order, method_name, decorators, sort_by):
        self.___objs = objs
        self.___compare = compare
        self.___ignore_order = ignore_order
        self.___method_name = method_name
        self.___decorators = decorators
        self.___sort_by = sort_by

    def _call(self, obj, args, kwargs):
        # copying the args and kwargs is important, because pymongo changes
        # the dicts (fits them with the _id)
        return self.___apply_decorators(
            getattr(obj, self.___method_name)(*_deepcopy(args), **_deepcopy(kwargs))
        )

    def _get_exception_type(self, obj, args, kwargs, name):
        try:
            self._call(obj, args, kwargs)
            assert False, "No exception raised for " + name
        except Exception as err:
            return type(err)

    def __call__(self, *args, **kwargs):
        if self.___compare == _COMPARE_EXCEPTIONS:
            results = dict(
                (name, self._get_exception_type(obj, args, kwargs, name=name))
                for name, obj in self.___objs.items()
            )
        else:
            results = dict(
                (name, self._call(obj, args, kwargs))
                for name, obj in self.___objs.items()
            )
        if self.___compare:
            _assert_no_diff(
                results, ignore_order=self.___ignore_order, sort_by=self.___sort_by
            )
        return results

    def ___apply_decorators(self, obj):
        for d in self.___decorators:
            obj = d(obj)
        return obj


def _assert_no_diff(results, ignore_order, sort_by):
    if _result_is_cursor(results) or _result_is_command_cursor(results):
        # If we were given a sort function, use that.
        if sort_by is not None:
            value_processor = functools.partial(
                _expand_cursor, sort=ignore_order, by=sort_by
            )
        else:
            value_processor = functools.partial(_expand_cursor, sort=ignore_order)
    else:
        assert not ignore_order
        value_processor = None
    prev_name = prev_value = None
    for index, (name, value) in enumerate(results.items()):
        if value_processor is not None:
            value = value_processor(value)
        if index > 0:
            d = diff(prev_value, value)
            assert not d, _format_diff_message(prev_name, name, d)
        prev_name = name
        prev_value = value


def _result_is_cursor(results):
    return any(type(result).__name__ == "Cursor" for result in results.values())


def _result_is_command_cursor(results):
    return any(type(result).__name__ == "CommandCursor" for result in results.values())


def by_id(document):
    return str(document.get("_id", str(document)))


def _expand_cursor(cursor, sort, by=by_id):
    returned = [result.copy() for result in cursor]
    if sort:
        returned.sort(key=by)
    for result in returned:
        result.pop("_id", None)
    return returned


def _format_diff_message(a_name, b_name, diff_list):
    msg = "Unexpected Diff:"
    for path, a_value, b_value in diff_list:
        a_path = [a_name] + path
        b_path = [b_name] + path
        msg += "\n\t{} != {} ({} != {})".format(
            ".".join(map(str, a_path)), ".".join(map(str, b_path)), a_value, b_value
        )
    return msg


def _deepcopy(x):
    """Deepcopy, but ignore regex objects..."""
    if isinstance(x, RE_TYPE):
        return x
    if isinstance(x, list) or isinstance(x, tuple):
        return type(x)(_deepcopy(y) for y in x)
    if isinstance(x, (dict, OrderedDict)):
        return type(x)((_deepcopy(k), _deepcopy(v)) for k, v in x.items())
    return copy.deepcopy(x)
