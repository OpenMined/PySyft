# third party
from tqdm import tqdm

# syft absolute
import syft as sy

# relative
from ...client.datasite_client import DatasiteClient
from ...client.syncing import compare_clients
from ...service.code.user_code import UserCode
from ...service.job.job_stash import Job
from ...service.job.job_stash import JobStatus
from ...service.request.request import Request
from ...service.request.request import RequestStatus
from ...service.sync.diff_state import ObjectDiffBatch
from ...types.result import Err


def deny_requests_without_autosync_tag(client_low: DatasiteClient):
    # Deny all requests that are not autosync
    requests = client_low.requests.get_all()
    if isinstance(requests, sy.SyftError):
        print(requests)
        return

    denied_requests = []
    for request in tqdm(requests):
        if request.status != RequestStatus.PENDING:
            continue
        if "autosync" not in request.tags:
            request.deny(
                reason="This request has been denied automatically. "
                "Please use the designated API to submit your request."
            )
            denied_requests.append(request.id)
    print(f"Denied {len(denied_requests)} requests without autosync tag")


def is_request_to_sync(batch: ObjectDiffBatch) -> bool:
    # True if this is a new low-side request
    # TODO add condition for sql requests/usercodes
    low_request = batch.root.low_obj
    return (
        isinstance(low_request, Request)
        and batch.status == "NEW"
        and "autosync" in low_request.tags
    )


def is_job_to_sync(batch: ObjectDiffBatch):
    # True if this is a new high-side job that is either COMPLETED or ERRORED
    if batch.status != "NEW":
        return False
    if not isinstance(batch.root.high_obj, Job):
        return False
    job = batch.root.high_obj
    return job.status in (JobStatus.ERRORED, JobStatus.COMPLETED)


def execute_requests(
    client_high: DatasiteClient, request_ids: list[sy.UID]
) -> dict[sy.UID, Job]:
    jobs_by_request_id = {}
    for request_id in request_ids:
        request = client_high.requests.get_by_uid(request_id)
        if not isinstance(request, Request):
            continue

        code = request.code
        if not isinstance(code, UserCode):
            continue

        func_name = request.code.service_func_name
        api_func = getattr(client_high.code, func_name, None)
        if api_func is None:
            continue

        job = api_func(blocking=False)
        jobs_by_request_id[request_id] = job

    return jobs_by_request_id


def deny_failed_jobs(
    client_low: DatasiteClient,
    jobs: list[Job],
) -> None:
    # NOTE no syncing is needed, requests are denied on the low side
    denied_requests = []

    for job in jobs:
        if job.status != JobStatus.ERRORED:
            continue

        error_result = job.result
        if isinstance(error_result, Err):
            error_msg = error_result.err_value
        else:
            error_msg = "An unknown error occurred, please check the Job logs for more information."

        code_id = job.user_code_id
        if code_id is None:
            continue
        requests = client_low.requests.get_by_usercode_id(code_id)
        if isinstance(requests, list) and len(requests) > 0:
            request = requests[0]
            request.deny(reason=f"Execution failed: {error_msg}")
            denied_requests.append(request.id)
        else:
            print(f"Failed to deny request for job {job.id}")

    print(f"Denied {len(denied_requests)} failed requests")


def sync_finished_jobs(
    client_low: DatasiteClient,
    client_high: DatasiteClient,
) -> dict[sy.UID, sy.SyftError | sy.SyftSuccess] | sy.SyftError:
    sync_job_results = {}
    synced_jobs = []
    diff = compare_clients(
        from_client=client_high, to_client=client_low, include_types=["job"]
    )
    if isinstance(diff, sy.SyftError):
        print(diff)
        return diff

    for batch in diff.batches:
        if is_job_to_sync(batch):
            job = batch.root.high_obj

            w = batch.resolve(build_state=False)
            share_result = w.click_share_all_private_data()
            if isinstance(share_result, sy.SyftError):
                sync_job_results[job.id] = share_result
                continue
            sync_result = w.click_sync()

            synced_jobs.append(job)
            sync_job_results[job.id] = sync_result

    print(f"Sharing {len(sync_job_results)} new results")
    deny_failed_jobs(client_low, synced_jobs)
    return sync_job_results


def sync_new_requests(
    client_low: DatasiteClient,
    client_high: DatasiteClient,
) -> dict[sy.UID, sy.SyftSuccess | sy.SyftError] | sy.SyftError:
    sync_request_results = {}
    diff = compare_clients(
        from_client=client_low, to_client=client_high, include_types=["request"]
    )
    if isinstance(diff, sy.SyftError):
        print(diff)
        return sync_request_results
    print(f"{len(diff.batches)} request batches found")
    for batch in tqdm(diff.batches):
        if is_request_to_sync(batch):
            request_id = batch.root.low_obj.id
            w = batch.resolve(build_state=False)
            result = w.click_sync()
            sync_request_results[request_id] = result
    return sync_request_results


def sync_and_execute_new_requests(
    client_low: DatasiteClient, client_high: DatasiteClient
) -> None:
    sync_results = sync_new_requests(client_low, client_high)
    if isinstance(sync_results, sy.SyftError):
        print(sync_results)
        return

    request_ids = [
        uid for uid, res in sync_results.items() if isinstance(res, sy.SyftSuccess)
    ]
    print(f"Synced {len(request_ids)} new requests")

    jobs_by_request = execute_requests(client_high, request_ids)
    print(f"Started {len(jobs_by_request)} new jobs")


def auto_sync(client_low: DatasiteClient, client_high: DatasiteClient) -> None:
    print("Starting auto sync")
    print("Denying non tagged jobs")
    deny_requests_without_autosync_tag(client_low)
    print("Syncing and executing")
    sync_and_execute_new_requests(client_low, client_high)
    sync_finished_jobs(client_low, client_high)
    print("Finished auto sync")
