# syft absolute
import syft
from syft.client.domain_client import DomainClient
from syft.client.syncing import compare_clients
from syft.client.syncing import resolve_single
from syft.service.job.job_stash import Job


def compare_and_resolve(*, from_client: DomainClient, to_client: DomainClient):
    diff_state_before = compare_clients(from_client, to_client)
    for obj_diff_batch in diff_state_before.batches:
        widget = resolve_single(obj_diff_batch)
        widget.click_share_all_private_data()
        widget.click_sync()
    from_client.refresh()
    to_client.refresh()
    diff_state_after = compare_clients(from_client, to_client)
    return diff_state_before, diff_state_after


def run_and_accept_result(client):
    job_high = client.code.compute(blocking=True)
    client.requests[0].accept_by_depositing_result(job_high)
    return job_high


@syft.syft_function_single_use()
def compute() -> int:
    return 42


def test_diff_widget_merge_status(low_worker, high_worker):
    low_client = low_worker.root_client
    client_low_ds = low_worker.guest_client
    high_client = high_worker.root_client

    _ = client_low_ds.code.request_code_execution(compute)

    diff_before, diff_after = compare_and_resolve(
        from_client=low_client, to_client=high_client
    )
    run_and_accept_result(high_client)
    diff_before, diff_after = compare_and_resolve(
        from_client=high_client, to_client=low_client
    )

    # expect to have a job diff, might fail if the implementation changes
    job_diff = [batch for batch in diff_after.batches if batch.root_type is Job][0]
    assert job_diff.status == "SAME", "Job diff status should be SAME after syncing"
