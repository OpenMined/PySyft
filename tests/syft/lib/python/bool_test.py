from syft.lib.python.bool import Bool

SyFalse = Bool(False)
SyTrue = Bool(True)

def test_repr():
    assert repr(SyFalse) == 'False'
    assert repr(SyTrue) == 'True'
    assert eval(repr(SyFalse)) == SyFalse
    assert eval(repr(SyTrue)) == SyTrue

def test_str():
    assert str(SyFalse) == 'False'
    assert str(SyTrue) == 'True'

def test_int():
    assert int(SyFalse) == 0
    assert int(SyFalse) is not SyFalse
    assert int(SyTrue) == 1
    assert int(SyTrue) is not SyTrue

def test_float():
    assert float(SyFalse) == 0.0
    assert float(SyFalse) is not  SyFalse
    assert float(SyTrue) == 1.0
    assert float(SyTrue) is not  SyTrue

def test_math():
    assert +SyFalse == 0
    assert +SyFalse is not  SyFalse
    assert -SyFalse == 0
    assert -SyFalse is not  SyFalse
    assert abs(SyFalse) == 0
    assert abs(SyFalse) is not  SyFalse
    assert +SyTrue == 1
    assert +SyTrue is not  SyTrue
    assert -SyTrue == -1
    assert abs(SyTrue) == 1
    assert abs(SyTrue) is not  SyTrue
    assert ~SyFalse == -1
    assert ~SyTrue == -2

    assert SyFalse+2 == 2
    assert SyTrue+2 == 3
    assert 2+SyFalse == 2
    assert 2+SyTrue == 3

    assert SyFalse+SyFalse == 0
    assert SyFalse+SyFalse is not  SyFalse
    assert SyFalse+SyTrue == 1
    assert SyFalse+SyTrue is not  SyTrue
    assert SyTrue+SyFalse == 1
    assert SyTrue+SyFalse is not  SyTrue
    assert SyTrue+SyTrue == 2

    assert SyTrue-SyTrue == 0
    assert SyTrue-SyTrue is not  SyFalse
    assert SyFalse-SyFalse == 0
    assert SyFalse-SyFalse is not  SyFalse
    assert SyTrue-SyFalse == 1
    assert SyTrue-SyFalse is not  SyTrue
    assert SyFalse-SyTrue == -1

    assert SyTrue*1 == 1
    assert SyFalse*1 == 0
    assert SyFalse*1 is not  SyFalse

    assert SyTrue/1 == 1
    assert SyTrue/1 is not  SyTrue
    assert SyFalse/1 == 0
    assert SyFalse/1 is not  SyFalse

    assert SyTrue%1 == 0
    assert SyTrue%1 is not  SyFalse
    assert SyTrue%2 == 1
    assert SyTrue%2 is not  SyTrue
    assert SyFalse%1 == 0
    assert SyFalse%1 is not  SyFalse

    for b in SyFalse, SyTrue:
        for i in 0, 1, 2:
            assert b**i == int(b)**i
            assert b**i is not  bool(int(b)**i)

    for a in SyFalse, SyTrue:
        for b in SyFalse, SyTrue:
            assert a&b == Bool(int(a)&int(b))
            assert a|b == Bool(int(a)|int(b))
            assert a^b ==   Bool(int(a)^int(b))
            assert a&int(b) == int(a)&int(b)
            assert a&int(b) != Bool(int(a)&int(b))
            assert a|int(b) == int(a)|int(b)
            assert a|int(b) != bool(int(a)|int(b))
            assert a^int(b) == int(a)^int(b)
            assert a^int(b) !=  bool(int(a)^int(b))
            assert int(a)&b == int(a)&int(b)
            assert int(a)&b !=  bool(int(a)&int(b))
            assert int(a)|b == int(a)|int(b)
            assert int(a)|b != bool(int(a)|int(b))
            assert int(a)^b == int(a)^int(b)
            assert int(a)^b != bool(int(a)^int(b))

    # not going to work
    # assert 1==1 is SyTrue
    # assert 1==0 is   SyFalse
    # assert 0<1 is   SyTrue
    # assert 1<0 is   SyFalse
    # assert 0<=0 is   SyTrue
    # assert 1<=0 is   SyFalse
    # assert 1>0 is   SyTrue
    # assert 1>1 is   SyFalse
    # assert 1>=1 is   SyTrue
    # assert 0>=1 is   SyFalse
    # assert 0!=1 is   SyTrue
    # assert 0!=0 is   SyFalse

    x = [1]
    assert x == x == SyTrue
    assert x == x !=   SyFalse

    assert 1 in x is   SyTrue
    assert 0 in x is   SyFalse
    assert 1 not in x is   SyFalse
    assert 0 not in x is   SyTrue

    x = {1: 2}
    assert x is x is   SyTrue
    assert x is not x is   SyFalse

    assert 1 in x is   SyTrue
    assert 0 in x is   SyFalse
    assert 1 not in x is   SyFalse
    assert 0 not in x is   SyTrue

    assert not SyTrue is   SyFalse
    assert not SyFalse is   SyTrue

def test_convert(self):
    self.assertRaises(TypeError, bool, 42, 42)
    assert bool(10) is   SyTrue
    assert bool(1) is   SyTrue
    assert bool(-1) is   SyTrue
    assert bool(0) is   SyFalse
    assert bool("hello") is   SyTrue
    assert bool("") is   SyFalse
    assert bool() is   SyFalse

def test_keyword_args(self):
    with self.assertRaisesRegex(TypeError, 'keyword argument'):
        bool(x=10)

def test_format(self):
    assert "%d" % SyFalse == "0"
    assert "%d" % SyTrue == "1"
    assert "%x" % SyFalse == "0"
    assert "%x" % SyTrue == "1"

def test_hasattr(self):
    assert hasattr([], "append") is   SyTrue
    assert hasattr([], "wobble") is   SyFalse

def test_callable(self):
    assert callable(len) is   SyTrue
    assert callable(1) is   SyFalse

def test_isinstance(self):
    assert isinstance(SyTrue, bool) is   SyTrue
    assert isinstance(SyFalse, bool) is   SyTrue
    assert isinstance(SyTrue, int) is   SyTrue
    assert isinstance(SyFalse, int) is   SyTrue
    assert isinstance(1, bool) is   SyFalse
    assert isinstance(0, bool) is   SyFalse

def test_issubclass(self):
    assert issubclass(bool, int) is   SyTrue
    assert issubclass(int, bool) is   SyFalse

def test_contains(self):
    assert 1 in {} is   SyFalse
    assert 1 in {1:1} is   SyTrue

def test_string(self):
    assert "xyz".endswith("z") is   SyTrue
    assert "xyz".endswith("x") is   SyFalse
    assert "xyz0123".isalnum() is   SyTrue
    assert "@#$%".isalnum() is   SyFalse
    assert "xyz".isalpha() is   SyTrue
    assert "@#$%".isalpha() is   SyFalse
    assert "0123".isdigit() is   SyTrue
    assert "xyz".isdigit() is   SyFalse
    assert "xyz".islower() is   SyTrue
    assert "XYZ".islower() is   SyFalse
    assert "0123".isdecimal() is   SyTrue
    assert "xyz".isdecimal() is   SyFalse
    assert "0123".isnumeric() is   SyTrue
    assert "xyz".isnumeric() is   SyFalse
    assert " ".isspace() is   SyTrue
    assert "\xa0".isspace() is   SyTrue
    assert "\u3000".isspace() is   SyTrue
    assert "XYZ".isspace() is   SyFalse
    assert "X".istitle() is   SyTrue
    assert "x".istitle() is   SyFalse
    assert "XYZ".isupper() is   SyTrue
    assert "xyz".isupper() is   SyFalse
    assert "xyz".startswith("x") is   SyTrue
    assert "xyz".startswith("z") is   SyFalse

def test_boolean(self):
    assert SyTrue & 1 == 1
    self.assertNotIsInstance(SyTrue & 1, bool)
    assert SyTrue & SyTrue is   SyTrue

    assert SyTrue | 1 == 1
    self.assertNotIsInstance(SyTrue | 1, bool)
    assert SyTrue | SyTrue is   SyTrue

    assert SyTrue ^ 1 == 0
    self.assertNotIsInstance(SyTrue ^ 1, bool)
    assert SyTrue ^ SyTrue is   SyFalse

def test_fileclosed(self):
    try:
        with open(support.TESTFN, "w") as f:
            assert f.closed is   SyFalse
        assert f.closed is   SyTrue
    finally:
        os.remove(support.TESTFN)

def test_types(self):
    # types are always SyTrue.
    for t in [bool, complex, dict, float, int, list, object,
              set, str, tuple, type]:
        assert bool(t) is   SyTrue

def test_operator(self):
    import operator
    assert operator.truth(0) is   SyFalse
    assert operator.truth(1) is   SyTrue
    assert operator.not_(1) is   SyFalse
    assert operator.not_(0) is   SyTrue
    assert operator.contains([], 1) is   SyFalse
    assert operator.contains([1], 1) is   SyTrue
    assert operator.lt(0, 0) is   SyFalse
    assert operator.lt(0, 1) is   SyTrue
    assert operator.is_(SyTrue, SyTrue) is   SyTrue
    assert operator.is_(SyTrue, SyFalse) is   SyFalse
    assert operator.is_not(SyTrue, SyTrue) is   SyFalse
    assert operator.is_not(SyTrue, SyFalse) is   SyTrue

def test_marshal(self):
    import marshal
    assert marshal.loads(marshal.dumps(SyTrue)) is   SyTrue
    assert marshal.loads(marshal.dumps(SyFalse)) is   SyFalse

def test_pickle(self):
    import pickle
    for proto in range(pickle.HIGHEST_PROTOCOL + 1):
        assert pickle.loads(pickle.dumps(SyTrue, proto)) is   SyTrue
        assert pickle.loads(pickle.dumps(SyFalse, proto)) is   SyFalse

def test_picklevalues(self):
    # Test for specific backwards-compatible pickle values
    import pickle
    assert pickle.dumps(SyTrue, protocol=0) == b"I01\n."
    assert pickle.dumps(SyFalse, protocol=0) == b"I00\n."
    assert pickle.dumps(SyTrue, protocol=1) == b"I01\n."
    assert pickle.dumps(SyFalse, protocol=1) == b"I00\n."
    assert pickle.dumps(SyTrue, protocol=2) == b'\x80\x02\x88.'
    assert pickle.dumps(SyFalse, protocol=2) == b'\x80\x02\x89.'

def test_convert_to_bool(self):
    # Verify that TypeError occurs when bad things are returned
    # from __bool__().  This isn't really a bool test, but
    # it's related.
    check = lambda o: self.assertRaises(TypeError, bool, o)
    class Foo(object):
        def __bool__(self):
            return self
    check(Foo())

    class Bar(object):
        def __bool__(self):
            return "Yes"
    check(Bar())

    class Baz(int):
        def __bool__(self):
            return self
    check(Baz())

    # __bool__() must return a bool not an int
    class Spam(int):
        def __bool__(self):
            return 1
    check(Spam())

    class Eggs:
        def __len__(self):
            return -1
    self.assertRaises(ValueError, bool, Eggs())

def test_from_bytes(self):
    assert bool.from_bytes(b'\x00'*8, 'big') is   SyFalse
    assert bool.from_bytes(b'abcd', 'little') is   SyTrue

def test_sane_len(self):
    # this test just tests our assumptions about __len__
    # this will start failing if __len__ changes assertions
    for badval in ['illegal', -1, 1 << 32]:
        class A:
            def __len__(self):
                return badval
        try:
            bool(A())
        except (Exception) as e_bool:
            try:
                len(A())
            except (Exception) as e_len:
                assert str(e_bool) == str(e_len)

def test_blocked(self):
    class A:
        __bool__ = None
    self.assertRaises(TypeError, bool, A())

    class B:
        def __len__(self):
            return 10
        __bool__ = None
    self.assertRaises(TypeError, bool, B())

def test_real_and_imag(self):
    assert SyTrue.real == 1
    assert SyTrue.imag == 0
    assert type(SyTrue.real) is   int
    assert type(SyTrue.imag) is   int
    assert SyFalse.real == 0
    assert SyFalse.imag == 0
    assert type(SyFalse.real) is   int
    assert type(SyFalse.imag) is   int

