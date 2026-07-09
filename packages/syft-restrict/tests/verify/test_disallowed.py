"""Things the private region is NOT allowed to do — default-deny rejects each one."""

import pytest

from syft_restrict import verify

from .conftest import error_codes


@pytest.mark.parametrize(
    "snippet",
    [
        "import os\n",
        "from os import path\n",
        "with ctx() as g:\n    pass\n",
        "try:\n    pass\nfinally:\n    pass\n",
        "raise ValueError()\n",
        "def f():\n    global x\n    x = 1\n",
        "def f():\n    y = 1\n    def g():\n        nonlocal y\n        y = 2\n    return g\n",
        "x = 1\ndel x\n",
        "assert x\n",
        "async def f():\n    return 1\n",
        "def f():\n    yield 1\n",
        "y = f'no interpolation'\n",  # banned outright, even with no {expr} -- drop the f-prefix
        "y = f'value={x}'\n",
    ],
)
def test_banned_constructs(verify_all, snippet):
    result = verify_all(snippet)
    assert "banned-construct" in error_codes(result), [
        (v.code, v.message) for v in result.violations
    ]


@pytest.mark.parametrize(
    "name",
    [
        "eval",
        "exec",
        "compile",
        "__import__",
        "getattr",
        "setattr",
        "delattr",
        "hasattr",
        "vars",
        "globals",
        "locals",
        "dir",
        "open",
        "input",
        "breakpoint",
        "memoryview",
    ],
)
def test_banned_calls(verify_all, name):
    result = verify_all(f"y = {name}(z)\n")
    assert "banned-call" in error_codes(result), [
        (v.code, v.message) for v in result.violations
    ]


@pytest.mark.parametrize(
    "snippet",
    [
        "a = x.reshape(8, -1)\n",
        "b = '{0.__class__}'.format(payload)\n",
        "c = items.append(1)\n",
    ],
)
def test_named_method_on_value(verify_all, snippet):
    assert "method-on-value" in error_codes(verify_all(snippet))


@pytest.mark.parametrize("snippet", ["a = x.shape\n", "b = x.T\n", "c = x.ndim\n"])
def test_attribute_read_on_value(verify_all, snippet):
    assert "attr-on-value" in error_codes(verify_all(snippet))


@pytest.mark.parametrize("snippet", ["c = obj.__class__\n", "d = obj.__dict__\n"])
def test_dunder_attribute_read(verify_all, snippet):
    assert "dunder-attr" in error_codes(verify_all(snippet))


@pytest.mark.parametrize("dunder", ["__init__", "__getattr__", "__reduce__"])
def test_disallowed_dunder_def(verify_all, dunder):
    src = f"class M(object):\n    def {dunder}(self):\n        return None\n"
    assert "dunder-def" in error_codes(verify_all(src))


@pytest.mark.parametrize(
    "snippet",
    [
        "@evil\ndef f():\n    return 1\n",
        # @property runs code on a bare attribute access — denied like any non-allow-listed decorator
        "class B(object):\n    @property\n    def w(self):\n        return 1\n",
    ],
)
def test_disallowed_decorator(verify_all, snippet):
    assert "decorator" in error_codes(verify_all(snippet))


def test_class_keyword_metaclass(verify_all):
    src = "class M(object, metaclass=Meta):\n    def __call__(self, x):\n        return x\n"
    assert "class-keyword" in error_codes(verify_all(src))


def test_non_allowlisted_base_class(verify_all):
    src = "class M(SomeLib):\n    def __call__(self, x):\n        return x\n"
    assert "class-base" in error_codes(verify_all(src))


def test_walrus_is_not_on_node_allowlist(verify_all):
    """NamedExpr is neither allowed nor explicitly banned -> node-type rejection."""
    assert "node-type" in error_codes(verify_all("y = (z := 1)\n"))


def test_reserved_module_alias_cannot_be_rebound(policy):
    source = "import jax.numpy as jnp\njnp = make_evil()\n"
    # only the rebind line is private (the import is public glue)
    result = verify(source, [[2, 2]], policy)
    assert "reserved-name" in error_codes(result)


def test_attribute_write_to_foreign_object(verify_all):
    """`some_obj.send = data` — only `self.<name>` writes are allowed; a foreign attribute
    target is an opaque value (exfiltration channel)."""
    assert "attr-on-value" in error_codes(verify_all("some_obj.send = data\n"))


def test_comprehension_over_io(verify_all):
    """`[v for v in open(f)]` reintroduces I/O through a denied call in the iterable."""
    assert "banned-call" in error_codes(verify_all("y = [v for v in open(f)]\n"))


def test_denied_call_in_passive_position(verify_all):
    """Call-checking is position-independent: a denied target in a default arg runs at
    def-creation time and is still caught."""
    src = "def f(a=eval('1')):\n    return a\n"
    assert "banned-call" in error_codes(verify_all(src))


def test_match_statement_is_not_on_node_allowlist(verify_all):
    """`match`/`case` is unlisted modern syntax -> default-deny node-type rejection."""
    src = "match x:\n    case 1:\n        y = 1\n"
    assert "node-type" in error_codes(verify_all(src))
