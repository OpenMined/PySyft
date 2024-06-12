# stdlib
from collections.abc import Callable
from collections.abc import Collection
from collections.abc import Generator
from collections.abc import Iterable
from collections.abc import Mapping
from functools import cached_property
from itertools import chain
from itertools import combinations
from typing import Any
from typing import Generic
from typing import TypeVar
import uuid

# third party
from faker import Faker
import pytest
from typing_extensions import Self

# syft absolute
from syft.service.dataset.dataset import Contributor
from syft.service.dataset.dataset import Dataset
from syft.service.dataset.dataset import DatasetPageView
from syft.service.user.roles import Roles
from syft.types.dicttuple import DictTuple


def test_dict_tuple_not_subclassing_mapping():
    assert not issubclass(DictTuple, Mapping)


# different ways to create a DictTuple
SIMPLE_TEST_CASES = [
    DictTuple({"x": 1, "y": 2}),
    DictTuple([("x", 1), ("y", 2)]),
    DictTuple([1, 2], ["x", "y"]),
]


@pytest.mark.parametrize("dict_tuple", SIMPLE_TEST_CASES)
def test_should_iter_over_value(dict_tuple: DictTuple) -> None:
    values = []
    for v in dict_tuple:
        values.append(v)  # noqa: PERF402

    assert values == [1, 2]


@pytest.mark.parametrize("dict_tuple", SIMPLE_TEST_CASES)
def test_indexing(dict_tuple: DictTuple) -> None:
    assert dict_tuple[0] == 1
    assert dict_tuple[1] == 2
    assert dict_tuple["x"] == 1
    assert dict_tuple["y"] == 2


@pytest.mark.parametrize("dict_tuple", SIMPLE_TEST_CASES)
def test_convert_to_other_iterable_types(dict_tuple: DictTuple) -> None:
    assert list(dict_tuple) == [1, 2]
    assert tuple(dict_tuple) == (1, 2)


@pytest.mark.parametrize("dict_tuple", SIMPLE_TEST_CASES)
def test_keys(dict_tuple) -> None:
    assert list(dict_tuple.keys()) == ["x", "y"]


@pytest.mark.parametrize("dict_tuple", SIMPLE_TEST_CASES)
def test_convert_to_dict(dict_tuple: DictTuple) -> None:
    assert dict(dict_tuple) == {"x": 1, "y": 2}


@pytest.mark.parametrize("dict_tuple", SIMPLE_TEST_CASES)
def test_convert_items_to_dicttest_get_mapping(dict_tuple: DictTuple) -> None:
    assert dict(dict_tuple.items()) == {"x": 1, "y": 2}


@pytest.mark.parametrize("dict_tuple", SIMPLE_TEST_CASES)
def test_iter_over_items(dict_tuple: DictTuple) -> None:
    items = []
    for k, v in dict_tuple.items():
        items.append((k, v))

    assert items == [("x", 1), ("y", 2)]


@pytest.mark.parametrize("dict_tuple", SIMPLE_TEST_CASES)
def test_dicttuple_is_not_a_mapping(dict_tuple: DictTuple) -> None:
    assert not isinstance(dict_tuple, Mapping)


_KT = TypeVar("_KT")
_VT = TypeVar("_VT")


class Case(Generic[_KT, _VT]):
    values: Collection[_VT]
    keys: Collection[_KT]
    key_fn: Callable[[_VT], _KT] | None
    value_generator: Callable[[], Generator[_VT, Any, None]]
    key_generator: Callable[[], Generator[_KT, Any, None]]

    def __init__(
        self,
        values: Collection[_VT],
        keys: Callable[[_VT], _KT] | Collection[_KT],
    ) -> None:
        self.values = values

        if isinstance(keys, Callable):
            self.key_fn = keys
            self.keys = [keys(v) for v in values]
        else:
            self.key_fn = None
            self.keys = keys

        def value_generator() -> Generator[_VT, Any, None]:
            yield from values

        def key_generator() -> Generator[_KT, Any, None]:
            yield from self.keys

        self.value_generator = value_generator
        self.key_generator = key_generator

    def kv(self) -> Iterable[tuple[_KT, _VT]]:
        return zip(self.keys, self.values)

    @cached_property
    def mapping(self) -> dict[_KT, _VT]:
        return dict(self.kv())

    def constructor_args(self, mapping: bool = True) -> list[Callable[[], tuple]]:
        return [
            lambda: (self.values, self.keys),
            lambda: (self.value_generator(), self.key_generator()),
            lambda: (self.values, self.key_generator()),
            lambda: (self.value_generator(), self.keys),
            *(
                [
                    lambda: (self.mapping,),
                    lambda: (self.kv(),),
                ]
                if mapping
                else []
            ),
            *(
                [
                    lambda: (self.values, self.key_fn),
                    lambda: (self.value_generator(), self.key_fn),
                ]
                if self.key_fn is not None
                else []
            ),
        ]

    def generate(self) -> Generator[DictTuple[_KT, _VT], Any, None]:
        return (DictTuple(*args()) for args in self.constructor_args())

    def generate_one(self) -> DictTuple[_KT, _VT]:
        return next(self.generate())

    @classmethod
    def from_kv(cls, kv: Mapping[_KT, _VT]) -> Self:
        return cls(kv.values(), kv.keys())

    def __repr__(self):
        return f"{self.__class__.__qualname__}{self.mapping}"


TEST_CASES: list[Case] = [
    Case(values=[1, 2, 3], keys=["x", "y", "z"]),
    Case(values=[1, 2, 3], keys=str),
]


@pytest.mark.parametrize(
    "args1,args2",
    chain.from_iterable(combinations(c.constructor_args(), 2) for c in TEST_CASES),
)
def test_all_equal(args1: Callable[[], tuple], args2: Callable[[], tuple]) -> None:
    d1 = DictTuple(*args1())
    d2 = DictTuple(*args2())

    assert d1 == d2
    assert d1.keys() == d2.keys()


@pytest.mark.parametrize(
    "dict_tuple,case",
    [(c.generate_one(), c) for c in TEST_CASES],
)
class TestDictTupleProperties:
    def test_should_iter_over_value(self, dict_tuple: DictTuple, case: Case) -> None:
        itered = (v for v in dict_tuple)
        assert all(a == b for a, b in zip(itered, case.values))

    def test_int_indexing(self, dict_tuple: DictTuple, case: Case) -> None:
        for i in range(len(dict_tuple)):
            assert dict_tuple[i] == case.values[i]

    def test_key_indexing(self, dict_tuple: DictTuple, case: Case) -> None:
        for k in case.keys:
            assert dict_tuple[k] == case.mapping[k]

    def test_convert_to_other_iterable_types(
        self, dict_tuple: DictTuple, case: Case
    ) -> None:
        assert list(dict_tuple) == list(case.values)
        assert tuple(dict_tuple) == tuple(case.values)

    def test_keys(self, dict_tuple: DictTuple, case: Case) -> None:
        assert list(dict_tuple.keys()) == list(case.keys)

    def test_dicttuple_is_not_a_mapping(
        self, dict_tuple: DictTuple, case: Case
    ) -> None:
        assert not isinstance(dict_tuple, Mapping)

    def test_convert_to_dict(self, dict_tuple: DictTuple, case: Case) -> None:
        assert dict(dict_tuple) == case.mapping

    def test_convert_items_to_dict(self, dict_tuple: DictTuple, case: Case) -> None:
        assert dict(dict_tuple.items()) == case.mapping

    def test_constructing_dicttuple_from_itself(
        self, dict_tuple: DictTuple, case: Case
    ) -> None:
        dd = DictTuple(dict_tuple)
        assert tuple(dd) == tuple(case.values)
        assert tuple(dd.keys()) == tuple(case.keys)


@pytest.mark.parametrize(
    "args", Case(values=["z", "b"], keys=[1, 2]).constructor_args()
)
def test_keys_should_not_be_int(args: Callable[[], tuple]) -> None:
    with pytest.raises(ValueError, match="int"):
        DictTuple(*args())


LENGTH_MISMACTH_TEST_CASES = [
    Case(values=[1, 2, 3], keys=["x", "y"]),
    Case(values=[1, 2], keys=["x", "y", "z"]),
]


@pytest.mark.parametrize(
    "args",
    chain.from_iterable(
        c.constructor_args(mapping=False) for c in LENGTH_MISMACTH_TEST_CASES
    ),
)
def test_keys_and_values_should_have_same_length(args: Callable[[], tuple]) -> None:
    with pytest.raises(ValueError, match="length"):
        DictTuple(*args())


def test_datasetpageview(faker: Faker):
    uploader = Contributor(
        name=faker.name(), role=str(Roles.UPLOADER), email=faker.email()
    )

    length = 10
    datasets = (
        Dataset(name=uuid.uuid4().hex, contributor={uploader}, uploader=uploader)
        for _ in range(length)
    )
    dict_tuple = DictTuple(datasets, lambda d: d.name)

    assert DatasetPageView(datasets=dict_tuple, total=length)


class EnhancedDictTuple(DictTuple):
    pass


@pytest.mark.parametrize(
    "args,case",
    chain.from_iterable(
        ((args, c) for args in c.constructor_args()) for c in TEST_CASES
    ),
)
def test_subclassing_dicttuple(args: Callable[[], tuple], case: Case):
    dict_tuple = DictTuple(*args())
    enhanced_dict_tuple = EnhancedDictTuple(*args())
    dict_tuple_enhanced_dict_tuple = DictTuple(enhanced_dict_tuple)
    enhanced_dict_tuple_dict_tuple = EnhancedDictTuple(dict_tuple)

    values = tuple(case.values)
    keys = tuple(case.keys)

    for d in (
        dict_tuple,
        enhanced_dict_tuple,
        dict_tuple_enhanced_dict_tuple,
        enhanced_dict_tuple_dict_tuple,
    ):
        assert tuple(d) == values
        assert tuple(d.keys()) == keys
