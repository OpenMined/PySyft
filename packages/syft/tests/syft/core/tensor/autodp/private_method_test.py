# third party
import numpy as np
import pytest

# syft absolute
import syft as sy
from syft.core.adp.entity import Entity


def test_incompatible_input_tensor_type() -> None:
    x = sy.Tensor(np.float32([1, 2, 3, 4.0]))
    with pytest.raises(TypeError, match=r".* wrapping np.int32 .*"):
        out = x.private(min_val=0, max_val=5, entities="bob")


def test_string_entity() -> None:
    x = sy.Tensor([1, 2, 3, 4])
    out = x.private(min_val=0, max_val=5, entities="bob")
    assert out.child.entity.name == "bob"


def test_list_of_strings_entity() -> None:
    x = sy.Tensor([1, 2, 3, 4])
    out = x.private(min_val=0, max_val=5, entities=["bob", "bob", "bob", "alice"])
    assert out.child.entities[0][0].name == "bob"
    assert out.child.entities[-1][0].name == "alice"


def test_class_entity() -> None:
    x = sy.Tensor([1, 2, 3, 4])
    out = x.private(min_val=0, max_val=5, entities=Entity("bob"))
    assert out.child.entity.name == "bob"


def test_list_of_entity_objs() -> None:
    x = sy.Tensor([1, 2, 3, 4])
    out = x.private(
        min_val=0,
        max_val=5,
        entities=[Entity("bob"), Entity("bob"), Entity("bob"), Entity("alice")],
    )
    assert out.child.entities[0][0].name == "bob"
    assert out.child.entities[-1][0].name == "alice"
