"""Things the hidden region IS allowed to do — these must verify cleanly."""

from syft_restrict import verify

from .conftest import FIXTURES, error_codes, make_policy


def _ok(result):
    return result.ok, [f"L{v.line} {v.code}: {v.message}" for v in result.violations]


def test_compliant_fixture_passes(policy):
    """The green-path fixture (model definition) passes with no violations."""
    source = (FIXTURES / "compliant_model.py").read_text()
    config_line = next(
        i for i, ln in enumerate(source.splitlines(), 1) if ln.startswith("CONFIG")
    )
    result = verify(source, [[config_line, len(source.splitlines())]], policy)
    ok, detail = _ok(result)
    assert ok, detail
    assert result.n_calls_checked > 0


def test_self_method_calls_allowed(verify_all):
    """``self.method(...)`` / ``cls.method(...)`` — receiver is the module class, not opaque."""
    src = (
        "class M:\n"
        "    def setup(self):\n"
        "        self.w = self.param('w')\n"
        "    def __call__(self, x):\n"
        "        return self.norm(x)\n"
    )
    assert _ok(verify_all(src))[0]


def test_bare_name_calls_allowed(verify_all):
    """Calling a local var / hidden def / safe builtin by bare name is fine."""
    src = (
        "def helper(n):\n"
        "    rows = list(range(n))\n"
        "    vals = tuple(rows)\n"
        "    total = sum(vals)\n"
        "    return helper(total)\n"
    )
    assert _ok(verify_all(src))[0]


def test_allowed_decorators(policy):
    """Only the allow-listed decorators may sit above a def/class (imports stay visible)."""
    header = "from flax import linen as nn\nimport jax\n"
    body = (
        "class M:\n"
        "    @nn.compact\n"
        "    @jax.jit\n"
        "    def __call__(self, x):\n"
        "        return x\n"
    )
    source = header + body
    result = verify(source, [[3, len(source.splitlines())]], policy)
    assert _ok(result)[0]


def test_allowed_dunder_defs(verify_all):
    """``__call__``, ``setup`` and ``__post_init__`` are the only definable hooks."""
    src = (
        "class M:\n"
        "    def setup(self):\n"
        "        return None\n"
        "    def __post_init__(self):\n"
        "        return None\n"
        "    def __call__(self, x):\n"
        "        return x\n"
    )
    assert _ok(verify_all(src))[0]


def test_class_base_must_be_allow_listed_import(verify_all):
    """A base class must be an allow-listed import; ``object`` and hidden-region classes are rejected
    (their metaclass/__init_subclass__ would run at class-creation time)."""
    ok_src = (
        "from flax import linen as nn\n"
        "class M(nn.Module):\n"
        "    def __call__(self, x):\n"
        "        return x\n"
    )
    result = verify(ok_src, [[2, len(ok_src.splitlines())]], make_policy())
    assert _ok(result)[0]

    for base in ("object", "Base"):
        src = (
            "class Base:\n"
            "    def __call__(self, x):\n"
            "        return x\n"
            f"class Child({base}):\n"
            "    def __call__(self, x):\n"
            "        return x\n"
        )
        assert "class-base" in error_codes(verify_all(src))


def test_control_flow_and_comprehensions(verify_all):
    """if/for/while/ternary, comprehensions and a lambda are all on the node allow-list."""
    src = (
        "def f(xs):\n"
        "    acc = [g(v) for v in xs if v]\n"
        "    pairs = {k: k for k in xs}\n"
        "    seen = {v for v in xs}\n"
        "    h = (lambda z: z)\n"
        "    out = 0\n"
        "    for v in xs:\n"
        "        out = v\n"
        "        if out:\n"
        "            break\n"
        "    while out:\n"
        "        out = 0\n"
        "    return out if xs else h(acc)\n"
    )
    assert _ok(verify_all(src))[0]


def test_fstring_over_opaque_value_is_rejected(verify_all):
    """f-strings (JoinedStr/FormattedValue) are allowed as a node type, but interpolating an
    opaque value invokes __format__ on it with no Call node -- must be rejected the same as
    f"{x!r}"/f"{x!s}"/f"{x=}" (see test_bypasses.py)."""
    src = "def f(x):\n    return f'value={x}'\n"
    assert "method-on-value" in error_codes(verify_all(src))


def test_calling_a_value_is_allowed(verify_all):
    """Calling the result of a subscript/call (its ``__call__``) is allowed."""
    src = (
        "class M:\n"
        "    def __call__(self, x):\n"
        "        layer = self.layers[0]\n"
        "        return self.layers[0](x) + layer(x)\n"
    )
    assert _ok(verify_all(src))[0]


def test_non_reserved_rebind_is_fine(verify_all):
    """Reassigning ordinary locals (not import aliases / wrapper names) is allowed."""
    src = "def f(x):\n    y = x\n    y += 1\n    z: int = y\n    return z\n"
    result = verify_all(src)
    assert _ok(result)[0]
    assert "reserved-name" not in error_codes(result)


def test_arithmetic_only_policy_still_passes_pure_math(verify_all):
    """A bundle the code doesn't use being disabled doesn't cause spurious failures."""
    pol = make_policy(methods=["arithmetic"])
    src = "def f(a, b):\n    return a + b - (-a)\n"
    assert _ok(verify_all(src, pol))[0]
