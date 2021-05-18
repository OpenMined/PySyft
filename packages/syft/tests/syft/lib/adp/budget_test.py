# syft absolute
import syft as sy
from syft.lib.adp.entity import Entity
from syft.lib.adp.scalar import Scalar


def test_domain_budget_method() -> None:
    domain = sy.Domain(name="alice")
    root_client = domain.get_root_client()

    budget = root_client.budget()
    assert str(budget) == f"{type(budget)}: {budget.value}"
    assert budget.value == 0


# def test_domain_budget_scalar_publish() -> None:
#     domain = sy.Domain(name="alice")
#     root_client = domain.get_root_client()
#
#     bob = Scalar(value=1, min_val=-2, max_val=2, entity=Entity(unique_name="Bob"))
#     alice = Scalar(value=1, min_val=-1, max_val=1, entity=Entity(unique_name="Alice"))
#     charlie = Scalar(value=2, min_val=-2, max_val=2, entity=Entity(unique_name="Charlie"))
#     david = Scalar(value=2, min_val=-2, max_val=2, entity=Entity(unique_name="David"))
#
#     out = bob * bob * bob + charlie + david + bob + alice
#
#     public_out = out.publish(acc=acc, sigma=0.5)
#
#     assert True is False
