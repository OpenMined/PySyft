"""Default-deny tests: things the private region is NOT allowed to do by default.

Each test is a straightforward rejection of a policy category (banned-construct,
banned-name, unresolved-call, etc.
"""

import pytest
from syft_restrict import verify

from verify.helpers import get_error_codes, normalize_source

# ── banned node types / constructs ───────────────────────────────────────────


@pytest.mark.parametrize(
    "snippet",
    [
        "import os",
        "from os import path",
        """
        with ctx() as g:
            pass
        """,
        """
        try:
            pass
        finally:
            pass
        """,
        "raise ValueError()",
        """
        def f():
            global x
            x = 1
        """,
        """
        def f():
            y = 1
            def g():
                nonlocal y
                y = 2
            return g
        """,
        """
        x = 1
        del x
        """,
        "assert x",
        """
        async def f():
            return 1
        """,
        """
        def f():
            yield 1
        """,
        "y = f'no interpolation'",  # banned outright, even with no {expr}
        "y = f'value={x}'",
    ],
)
def test_banned_constructs(verify_all, snippet):
    result = verify_all(snippet)
    assert "banned-construct" in get_error_codes(result), [
        (v.code, v.message) for v in result.violations
    ]


def test_walrus_is_not_on_node_allowlist(verify_all):
    """NamedExpr is neither allowed nor explicitly banned -> node-type rejection."""
    assert "node-type" in get_error_codes(verify_all("y = (z := 1)"))


def test_match_statement_is_not_on_node_allowlist(verify_all):
    """``match``/``case`` is unlisted modern syntax -> default-deny node-type rejection."""
    src = """
    match x:
        case 1:
            y = 1
    """
    assert "node-type" in get_error_codes(verify_all(src))


# ── banned names (any Load reference) ────────────────────────────────────────


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
        "type",
        "__build_class__",
        "print",
        "repr",
        "str",
        "ascii",
        "format",
        "bytes",
        # site-injected builtins: stdout channels (copyright/credits/license),
        # interpreter shutdown (exit/quit), interactive help (help)
        "copyright",
        "credits",
        "license",
        "exit",
        "quit",
        "help",
    ],
)
def test_banned_names(verify_all, name):
    result = verify_all(f"y = {name}")
    assert "banned-name" in get_error_codes(result), [
        (v.code, v.message) for v in result.violations
    ]


def test_banned_call_reports_banned_name_once(verify_all):
    """A bare call to a banned builtin must not also emit call-unresolved."""
    result = verify_all("open('/etc/passwd')")
    codes = get_error_codes(result)
    assert codes == {"banned-name"}, codes


def test_comprehension_over_io(verify_all):
    """``[v for v in open(f)]`` reintroduces I/O through a denied call in the iterable."""
    assert "banned-name" in get_error_codes(verify_all("y = [v for v in open(f)]"))


def test_denied_call_in_passive_position(verify_all):
    """Call-checking is position-independent: a denied target in a default arg runs at
    def-creation time and is still caught."""
    src = """
    def f(a=eval('1')):
        return a
    """
    assert "banned-name" in get_error_codes(verify_all(src))


# ── attributes / methods on opaque values ────────────────────────────────────


@pytest.mark.parametrize(
    "snippet",
    [
        "a = x.reshape(8, -1)",
        "b = '{0.__class__}'.format(payload)",
        "c = items.append(1)",
    ],
)
def test_named_method_on_value(verify_all, snippet):
    assert "method-on-value" in get_error_codes(verify_all(snippet))


@pytest.mark.parametrize("snippet", ["a = x.shape", "b = x.T", "c = x.ndim"])
def test_attribute_read_on_value(verify_all, snippet):
    assert "attr-on-value" in get_error_codes(verify_all(snippet))


def test_attribute_write_to_foreign_object(verify_all):
    """``some_obj.send = data`` — only ``self.<name>`` writes are allowed."""
    assert "attr-on-value" in get_error_codes(verify_all("some_obj.send = data"))


@pytest.mark.parametrize("snippet", ["c = obj.__class__", "d = obj.__dict__"])
def test_dunder_attribute_read(verify_all, snippet):
    assert "dunder-attr" in get_error_codes(verify_all(snippet))


def test_bare_class_dunder_name_is_denied(verify_all):
    """``__class__`` as a bare Name (not Attribute) is still a dunder surface."""
    src = """
    class M:
        def __call__(self, x):
            c = __class__
            return x
    """
    assert "dunder-name" in get_error_codes(verify_all(src))


# ── defs / classes / decorators ──────────────────────────────────────────────


@pytest.mark.parametrize("dunder", ["__init__", "__getattr__", "__reduce__"])
def test_disallowed_dunder_def(verify_all, dunder):
    src = f"""
    class M:
        def {dunder}(self):
            return None
    """
    assert "dunder-def" in get_error_codes(verify_all(src))


@pytest.mark.parametrize(
    "snippet",
    [
        """
        @evil
        def f():
            return 1
        """,
        """
        class B:
            @property
            def w(self):
                return 1
        """,
        """
        class B:
            @staticmethod
            def w():
                return 1
        """,
    ],
)
def test_disallowed_decorator(verify_all, snippet):
    assert "banned-construct" in get_error_codes(verify_all(snippet))


@pytest.mark.parametrize("decorator", ["nn.compact", "jax.jit"])
def test_disallowed_decorator_even_when_otherwise_allowed(verify_all, decorator):
    # Decorators are banned outright, regardless of what they resolve to -- even a decorator that
    # would otherwise be a perfectly allow-listed call (nn.compact, jax.jit) is still rejected.
    src = f"""
    import jax
    from flax import linen as nn

    class B(nn.Module):
        @{decorator}
        def w(self):
            return 1
    """
    result = verify_all(src, private=[[4, len(normalize_source(src).splitlines())]])
    assert "banned-construct" in get_error_codes(result)


def test_class_keyword_metaclass(verify_all):
    src = """
    class M(metaclass=Meta):
        def __call__(self, x):
            return x
    """
    assert "class-keyword" in get_error_codes(verify_all(src))


def test_non_allowlisted_base_class(verify_all):
    src = """
    class M(SomeLib):
        def __call__(self, x):
            return x
    """
    assert "class-base" in get_error_codes(verify_all(src))


@pytest.mark.parametrize("base", ["object", "Base"])
def test_object_and_private_bases_are_denied(verify_all, base):
    """``object`` and private-region classes are rejected as bases (metaclass /
    __init_subclass__ would run at class-creation time)."""
    src = f"""
    class Base:
        def __call__(self, x):
            return x
    class Child({base}):
        def __call__(self, x):
            return x
    """
    assert "class-base" in get_error_codes(verify_all(src))


# ── reserved names / trusted identifiers ─────────────────────────────────────


def test_reserved_module_alias_cannot_be_rebound(policy):
    # only the rebind line is private (the import is public glue)
    source = normalize_source("""
    import jax.numpy as jnp
    jnp = make_evil()
    """)
    result = verify(source, [[2, 2]], policy)
    assert "reserved-name" in get_error_codes(result)


def test_safe_builtin_name_cannot_be_rebound(verify_all):
    """Bare calls to ``list``/``range``/… are trusted by identifier; shadowing
    must be denied."""
    src = """
    def helper():
        list = None
        return list
    """
    assert "reserved-name" in get_error_codes(verify_all(src))


# ── call-target default-deny ─────────────────────────────────────────────────


def test_call_through_bare_parameter_is_rejected(verify_all):
    """A parameter is never a provably safe callee."""
    src = """
    def apply(fn, x):
        return fn(x)
    """
    assert "call-unresolved" in get_error_codes(verify_all(src))


def test_call_through_unresolvable_name_is_rejected(verify_all):
    """A bare name that isn't an import, def/class, safe builtin, or tracked-safe
    local has no provenance."""
    src = """
    def f(x):
        return g(x)
    """
    assert "call-unresolved" in get_error_codes(verify_all(src))


def test_chained_call_through_unresolved_value_is_rejected(verify_all):
    """``d['k'](x)``: callee is a subscript on an arbitrary parameter."""
    src = """
    def helper(d, x):
        return d['k'](x)
    """
    assert "call-unresolved" in get_error_codes(verify_all(src))


# ── star imports (banned anywhere, even in the trusted public region) ─────────


def test_star_import_in_public_is_blocked(verify_all):
    # `from x import *` can silently shadow a name the private region trusts by spelling (a safe
    # builtin, import alias, wrapper) and is unreviewable, so it is banned even in public code.
    src = """
    from evil import *
    def f(x):
        return len(x)
    """
    assert "star-import" in get_error_codes(verify_all(src, private=[[2, 3]]))


def test_star_import_in_private_is_blocked(verify_all):
    assert "star-import" in get_error_codes(verify_all("from os import *"))
