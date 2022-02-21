# third party
import pytest

# syft absolute
import syft as sy


# MADHAVA: this needs fixing
@pytest.mark.xfail
def test_domain_budget_method(domain: sy.Domain) -> None:
    root_client = domain.get_root_client()

    budget = root_client.budget()
    assert str(budget) == f"{type(budget)}: {budget.value}"
    assert budget.value == 0
