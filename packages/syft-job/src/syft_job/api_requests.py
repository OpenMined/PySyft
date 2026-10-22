"""Turning an api call (a request) into an approved job.

A DS calls an api by writing a request to requests/<ds>/v1/<name>/: the api's
name in config.yaml, and the call's params in code/. When the DO's client
receives it, it checks it against the api and writes an ordinary job to
inbox/<ds>/v1/<name>/: the api's code plus the request's files. Creating the api
was the DO's approval in advance, so the job is approved at once. The request
folder is then removed. A request that fails a check is rejected and kept.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from .job import JobInfo
from .job_ref import JobRef, RequestRef
from .models import JobState, JobStatus
from .policies import PolicyResolutionError, ResolvedApi
from .submission import (
    REQUEST_FILE_NAME,
    PERMISSION_FILE_NAME,
    OwnerJobRecord,
    copy_submission,
    lock_for_submitter,
    submission_hash,
    validate_request,
)

if TYPE_CHECKING:
    from .client import JobClient

# The approval method of a job made from a request.
API_APPROVAL_METHOD = "api"


def receive_request(client: JobClient, ref: RequestRef) -> None:
    """Turn a request into an approved job in inbox/, or reject it.

    Waits while files are still arriving, and when the client can't load apis.
    """
    if _is_settled(client, ref) or client.api_resolver is None:
        return
    if not client._request_has_all_files(ref):
        return
    try:
        api = _checked_api(client, ref)
    except PolicyResolutionError as e:
        _reject(client, ref, str(e))
        return
    if _create_job(client, ref, api):
        _approve(client, ref.job_ref, api)
        shutil.rmtree(client.manager.request_dir(ref), ignore_errors=True)


def _is_settled(client: JobClient, ref: RequestRef) -> bool:
    """Whether the name already has a state: the request was rejected, turned
    into a job, or a whole job holds the name.

    A request folder left over after it became a job is removed. A request
    whose name a whole job holds is locked and left alone: review/ is keyed on
    the name, so it can't get a state of its own.
    """
    if not (client.manager.review_dir(ref.job_ref) / "state.yaml").exists():
        return False
    request_dir = client.manager.request_dir(ref)
    job_record = client.manager.read_owner_record(ref.job_ref)
    if job_record is not None and job_record.api_name:
        shutil.rmtree(request_dir, ignore_errors=True)
    elif not (request_dir / PERMISSION_FILE_NAME).exists():
        lock_for_submitter(request_dir, ref.ds_email)
    return True


def _checked_api(client: JobClient, ref: RequestRef) -> ResolvedApi:
    """The api the request calls, once the request passed every check.

    Raises PolicyResolutionError with the reason it is rejected.
    """
    request_dir = client.manager.request_dir(ref)
    valid, reason = validate_request(request_dir)
    if not valid:
        raise PolicyResolutionError(reason)
    if client.manager.submission_dir(ref.job_ref).exists():
        raise PolicyResolutionError("a job in inbox/ has the same name")
    api_name = client.manager.read_request(ref).api
    client.api_resolver.check_request(api_name, request_dir, ref.ds_email)
    return client.api_resolver(api_name)


def _reject(client: JobClient, ref: RequestRef, reason: str) -> None:
    """Lock the request, and give the job it would become a rejected state."""
    lock_for_submitter(client.manager.request_dir(ref), ref.ds_email)
    client.manager.write_state(
        ref.job_ref, client._rejected_state(datetime.now(timezone.utc), reason)
    )


def _create_job(client: JobClient, ref: RequestRef, api: ResolvedApi) -> bool:
    """Write the job to inbox/: the call's files, the api's code, config.yaml.

    The job is built in staging/, which only the DO reads, locked, and then
    moved into place in one step, so the DS never sees half a job or writes
    into it. Returns False when another process created it first.
    """
    job_ref = ref.job_ref
    building = client.manager.staging_dir(job_ref).parent / f".{ref.job_name}.building"
    _build_job(client, ref, api, building)
    job_dir = client.manager.submission_dir(job_ref)
    job_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.rename(building, job_dir)
    except OSError:
        shutil.rmtree(building, ignore_errors=True)
        return False
    return True


def _build_job(
    client: JobClient, ref: RequestRef, api: ResolvedApi, building: Path
) -> None:
    """Fill building with the job: the call's files, the api's code, its config.yaml."""
    shutil.rmtree(building, ignore_errors=True)
    copy_submission(client.manager.request_dir(ref), building)
    (building / REQUEST_FILE_NAME).unlink()
    copy_submission(api.files_dir, building)
    request = client.manager.read_request(ref)
    job_metadata = request.to_job_metadata(
        ref.job_name, ref.ds_email, ref.datasite_email
    )
    client.manager.write_submission(ref.job_ref, job_metadata, folder=building)
    lock_for_submitter(building, ref.ds_email)


def _approve(client: JobClient, job_ref: JobRef, api: ResolvedApi) -> None:
    """Record the job as received from the api, and approve it with the
    api's disclosures."""
    now = datetime.now(timezone.utc)
    record = OwnerJobRecord(
        received_hash=submission_hash(client.manager.submission_dir(job_ref)),
        received_at=now,
        api_name=api.name,
    )
    if not client.manager.claim_owner_record(job_ref, record):
        return
    state = JobState(status=JobStatus.PENDING, received_at=now)
    client.manager.write_state(job_ref, state)
    job = JobInfo(
        job_metadata=client.manager.read_submission(job_ref),
        state=state,
        current_user_email=client.current_user_email,
        client=client,
        ref=job_ref,
    )
    job.approve(approval_method=API_APPROVAL_METHOD, disclosures=api.disclosures)
