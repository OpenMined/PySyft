# stdlib
from time import time
from typing import Callable
from typing import Optional

# third party
from pydantic import BaseModel

# syft absolute
import syft as sy
from syft.serde.serializable import serializable


def get_fqn_for_class(cls):
    return f"{cls.__module__}.{cls.__name__}"


# ------------------------------ Simple classes ------------------------------


class AbstractBase:
    uid: str


@serializable(attrs=["uid", "value"])
class Base(AbstractBase):
    """Serialize: uid, value"""

    value: int

    def __init__(self, uid: str, value: int):
        self.uid = uid
        self.value = value


@serializable(attrs=["status"])
class Derived(Base):
    """Serialize: uid, value, status"""

    status: int

    def __init__(self, uid: str, value: int, status: int) -> None:
        super().__init__(uid, value)
        self.status = status


@serializable(attrs=["status"], without=["uid"])
class DerivedWithoutAttrs(Base):
    """Serialize: value, status"""

    status: int

    def __init__(self, uid: str, value: int, status: int) -> None:
        super().__init__(uid, value)
        self.status = status


@serializable(attrs=["status"], inherit=False)
class DerivedNoInherit(Base):
    """Serialize: status"""

    status: int

    def __init__(self, uid: str, value: int, status: int) -> None:
        super().__init__(uid, value)
        self.status = status


@serializable(attrs=["uid", "value"], inheritable=False)
class BaseAttrsNonInheritable(AbstractBase):
    """Serialize: uid, value (Derived cannot inherit base attrs)"""

    value: Optional[int]

    def __init__(self, uid: str = None, value: int = None):
        self.uid = uid
        self.value = value


@serializable(attrs=["status"])
class DerivedWithoutBaseAttrs(BaseAttrsNonInheritable):
    """Serialize: status (Dervied cannot inherit base attrs)"""

    status: int

    def __init__(self, uid: str, value: int, status: int):
        super().__init__(uid, value)

        self.uid = uid
        self.value = value
        self.status = status


def test_base_attrs():
    data = Base(uid=str(time()), value=2)

    ser = sy.serialize(data, to_bytes=True)
    de = sy.deserialize(ser, from_bytes=True)

    assert "uid" in data.__syft_serializable__
    assert "value" in data.__syft_serializable__

    assert (data.uid, data.value) == (
        de.uid,
        de.value,
    ), "Deserialized values do not match"


def test_base_non_inheritable_attrs():
    data = BaseAttrsNonInheritable(uid=str(time()), value=2)

    ser = sy.serialize(data, to_bytes=True)
    sy.deserialize(ser, from_bytes=True)

    assert "__syft_serializable__" not in data.__dict__


def test_derived():
    data = Derived(uid=str(time()), value=2, status=1)

    ser = sy.serialize(data, to_bytes=True)
    de = sy.deserialize(ser, from_bytes=True)

    assert "uid" in data.__syft_serializable__
    assert "value" in data.__syft_serializable__

    assert (data.uid, data.value, data.status) == (
        de.uid,
        de.value,
        de.status,
    ), "Deserialized values do not match"


def test_derived_without_attrs():
    data = DerivedWithoutAttrs(uid=str(time()), value=2, status=1)

    ser = sy.serialize(data, to_bytes=True)
    sy.deserialize(ser, from_bytes=True)

    assert "uid" not in data.__syft_serializable__
    assert "value" in data.__syft_serializable__
    assert "status" in data.__syft_serializable__


def test_derived_without_inherit():
    data = DerivedNoInherit(uid=str(time()), value=2, status=1)

    ser = sy.serialize(data, to_bytes=True)
    de = sy.deserialize(ser, from_bytes=True)

    assert "uid" not in data.__syft_serializable__
    assert "value" not in data.__syft_serializable__
    assert de.status == data.status


def test_derived_without_base_attrs():
    data = DerivedWithoutBaseAttrs(uid=str(time()), value=2, status=1)

    ser = sy.serialize(data, to_bytes=True)
    de = sy.deserialize(ser, from_bytes=True)

    assert "uid" not in data.__syft_serializable__
    assert "value" not in data.__syft_serializable__
    assert "status" in data.__syft_serializable__

    assert de.status == data.status


# ------------------------------ Pydantic classes ------------------------------


@serializable()
class PydBase(BaseModel):
    """Serialize: uid, value, flag"""

    uid: Optional[str] = None
    value: Optional[int] = None
    flag: Optional[bool] = None


@serializable()
class PydDerived(PydBase):
    """Serialize: uid, value, flag, source, target"""

    source: str
    target: str


@serializable(without=["uid"])
class PydDerivedWithoutAttr(PydBase):
    """
    Serialize: value, flag, source, target
    `without=` will only work with Optional attributes due to pydantic's validation
    """

    source: str
    target: str


@serializable(without=["uid", "flag", "config"])
class PydDerivedWithoutAttrs(PydBase):
    """
    Serialize: value, source, target
    `without=` will only work with Optional attributes due to pydantic's validation
    """

    source: str
    target: str
    config: Optional[dict] = None


@serializable(attrs=["source", "target"])
class PydDerivedOnly(PydBase):
    """
    Serialize: source, target
    """

    source: str
    target: str
    callback: Optional[Callable] = lambda: None  # noqa: E731


def test_pydantic():
    data = PydBase(uid=str(time()), value=2, flag=True)

    ser = sy.serialize(data, to_bytes=True)
    de = sy.deserialize(ser, from_bytes=True)

    assert (data.uid, data.value, data.flag) == (de.uid, de.value, de.flag)


def test_pydantic_derived():
    data = PydDerived(
        uid=str(time()),
        value=2,
        source="source_path",
        target="target_path",
    )

    ser = sy.serialize(data, to_bytes=True)
    de = sy.deserialize(ser, from_bytes=True)

    assert (data.uid, data.value, data.flag, data.source, data.target) == (
        de.uid,
        de.value,
        de.flag,
        de.source,
        de.target,
    )


def test_pydantic_derived_without_attr():
    data = PydDerivedWithoutAttr(
        uid=str(time()),
        value=2,
        source="source_path",
        target="target_path",
    )

    ser = sy.serialize(data, to_bytes=True)
    de = sy.deserialize(ser, from_bytes=True)

    assert data.uid is not None
    assert de.uid is None
    assert (data.value, data.flag, data.source, data.target) == (
        de.value,
        de.flag,
        de.source,
        de.target,
    )


def test_pydantic_derived_without_attrs():
    data = PydDerivedWithoutAttrs(
        uid=str(time()),
        value=2,
        source="source_path",
        target="target_path",
    )

    ser = sy.serialize(data, to_bytes=True)
    de = sy.deserialize(ser, from_bytes=True)

    assert (data.uid, data.flag, data.config) != (None, None, None)
    assert (de.uid, de.flag, de.config) == (None, None, None)
    assert (data.value, data.flag, data.source, data.target) == (
        de.value,
        de.flag,
        de.source,
        de.target,
    )


def test_pydantic_derived_only():
    data = PydDerivedOnly(
        uid=str(time()),
        value=2,
        flag=True,
        source="source_path",
        target="target_path",
    )

    ser = sy.serialize(data, to_bytes=True)
    de = sy.deserialize(ser, from_bytes=True)

    assert (data.uid, data.value, data.flag) != (de.uid, de.value, de.flag)
    assert (de.uid, de.value, de.flag) == (None, None, None)
    assert (data.source, data.target) == (de.source, de.target)
