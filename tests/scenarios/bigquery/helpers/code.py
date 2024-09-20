# stdlib
import asyncio

# third party
from helpers.api import api_for_path
from helpers.events import unsync

# syft absolute
from syft.service.code.user_code import UserCode
from syft.service.job.job_stash import Job


def get_approved(client):
    results = []
    for request in client.requests:
        if str(request.status) == "RequestStatus.APPROVED":
            results.append(request)
    return results


def run_code(client, method_name, **kwargs):
    service_func_name = method_name
    if "*" in method_name:
        matcher = method_name.replace("*", "")
        all_code = client.api.services.code.get_all()
        for code in all_code:
            if matcher in code.service_func_name:
                service_func_name = code.service_func_name
                break

    api_method = api_for_path(client, path=f"code.{service_func_name}")
    # can raise
    result = api_method(**kwargs)
    return result


def approve_and_deposit(client, request_id):
    request = client.requests.get_by_uid(uid=request_id)
    code = request.code

    if not isinstance(code, UserCode):
        return

    func_name = request.code.service_func_name
    job = run_code(client, func_name, blocking=False)
    if not isinstance(job, Job):
        return None

    job.wait()
    job_info = job.info(result=True)
    result = request.deposit_result(job_info, approve=True)
    return result


def get_pending(client):
    results = []
    for request in client.requests:
        if str(request.status) == "RequestStatus.PENDING":
            results.append(request)
    return results


@unsync
async def triage_requests(events, client, after, register, sleep=2):
    if after:
        await events.await_for(event_name=after)
    while True:
        await asyncio.sleep(sleep)
        requests = get_pending(client)
        for request in requests:
            approve_and_deposit(client, request.id)
            events.register(event_name=register)


@unsync
async def get_results(events, client, method_name, after, register):
    method_name = method_name.replace("*", "")
    if after:
        await events.await_for(event_name=after)
    while True:
        await asyncio.sleep(1)
        requests = get_approved(client)
        for request in requests:
            if method_name in request.code.service_func_name:
                job = run_code(client, request.code.service_func_name, blocking=False)
                if not isinstance(job, Job):
                    continue
                else:
                    result = job.wait().get()
                    if hasattr(result, "__len__") and len(result) == 10000:
                        events.register(event_name=register)
