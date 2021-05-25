# syft absolute
import syft as sy


def test_domain_budget_method() -> None:
    domain = sy.Domain(name="alice")
    root_client = domain.get_root_client()

    budget = root_client.budget()
    assert str(budget) == f"{type(budget)}: {budget.value}"
    assert budget.value == 0
