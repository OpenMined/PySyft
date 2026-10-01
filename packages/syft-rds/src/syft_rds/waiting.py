"""Find a job or a dataset by name, and wait until it is there.

The waits sync the client, then read the local job or dataset list, at each
poll. The loop is ``syft.sync.utils.waiting.wait_for``.
"""

from collections.abc import Callable, Iterable
from typing import Any, Optional

from syft.sync.utils.waiting import wait_for
from syft_job.models import JobStatus

# A job with one of these statuses does not change status again.
FINAL_JOB_STATUSES = frozenset(
    {JobStatus.DONE.value, JobStatus.FAILED.value, JobStatus.REJECTED.value}
)

JobStatusArg = str | JobStatus | Iterable[str | JobStatus] | None

# An extra condition on a job, for example ``lambda job: job.can_approve``.
JobCheck = Callable[[Any], bool]


class JobEndedError(RuntimeError):
    """The job reached a final status that the caller did not wait for."""


def job_statuses(status: JobStatusArg) -> Optional[frozenset[str]]:
    """The status values in ``status``, or None for "any status".

    Raises:
        ValueError: a value is not a ``JobStatus``.
    """
    if status is None:
        return None
    items = [status] if isinstance(status, (str, JobStatus)) else list(status)
    return frozenset(JobStatus(item).value for item in items)


def _jobs_named(jobs: Iterable[Any], job_name: str, user_name: Optional[str]) -> list:
    """Jobs called ``job_name``; with ``user_name``, only the ones it submitted or hosts."""
    return [
        job
        for job in jobs
        if job.name == job_name
        and (
            user_name is None
            or user_name in (job.submitted_by, job.datasite_owner_email)
        )
    ]


def _has_status(job: Any, statuses: Optional[frozenset[str]]) -> bool:
    """True when ``job`` has one of ``statuses``, or is done after one of them.

    Every status that is not final comes before ``done``. A job that is done
    therefore passed any of them, also if no poll saw it.
    """
    if statuses is None or job.status in statuses:
        return True
    return job.status == JobStatus.DONE.value and bool(statuses - FINAL_JOB_STATUSES)


def _single_job(
    jobs: Iterable[Any], job_name: str, user_name: Optional[str]
) -> Optional[Any]:
    """The job called ``job_name``, or None.

    Jobs with the name, the submitter and the datasite in common are runs of
    the same job, for example from an earlier run of a notebook. The newest
    run is the one returned.

    Raises:
        ValueError: jobs with the name have more than one submitter or datasite.
    """
    matches = _jobs_named(jobs, job_name, user_name)
    origins = sorted({(job.submitted_by, job.datasite_owner_email) for job in matches})
    if len(origins) > 1:
        candidates = ", ".join(
            f"submitted by {submitter} to {datasite}" for submitter, datasite in origins
        )
        raise ValueError(
            f"{len(matches)} jobs are named {job_name!r} ({candidates}). "
            "Pass user_name= to pick one."
        )
    # submitted_at is an ISO 8601 string, so the newest sorts last.
    return max(matches, key=lambda job: job.submitted_at or "", default=None)


def find_job(
    jobs: Iterable[Any],
    job_name: str,
    user_name: Optional[str],
    statuses: Optional[frozenset[str]],
    where: Optional[JobCheck] = None,
) -> Optional[Any]:
    """Return the job called ``job_name`` once it has one of ``statuses``.

    ``user_name`` matches the submitter or the datasite owner of the job. It is
    needed only when more than one job has the name. With ``where``, the job
    must also make ``where(job)`` True. Returns None while no job matches, or
    while the job can still change.

    Raises:
        ValueError: more than one job matches.
        JobEndedError: the job has a final status that is not in ``statuses``.
    """
    job = _single_job(jobs, job_name, user_name)
    if job is None:
        return None
    has_status = _has_status(job, statuses)
    if has_status and (where is None or where(job)):
        return job
    if not has_status and job.status in FINAL_JOB_STATUSES:
        raise JobEndedError(
            f"Job {job_name!r} ended as {job.status!r}, not "
            f"{' or '.join(sorted(statuses))}."
        )
    return None


def describe_job(
    jobs: Iterable[Any],
    job_name: str,
    user_name: Optional[str],
    statuses: Optional[frozenset[str]] = None,
    where: Optional[JobCheck] = None,
) -> str:
    """Say what the local job list holds for ``job_name`` now."""
    job = _single_job(jobs, job_name, user_name)
    if job is None:
        for_user = f" for {user_name}" if user_name else ""
        return f"no job named {job_name!r}{for_user} yet"
    text = f"job {job_name!r} is {job.status!r}"
    has_status = _has_status(job, statuses)
    if where is not None and has_status and not where(job):
        text += ", and the where= check is False"
    return text


def find_dataset(
    datasets: Iterable[Any], name: str, datasite: Optional[str]
) -> Optional[Any]:
    """Return the dataset called ``name``, owned by ``datasite`` when given.

    Raises:
        ValueError: more than one owner has a dataset called ``name``.
    """
    matches = [
        dataset
        for dataset in datasets
        if dataset.name == name and (datasite is None or dataset.owner == datasite)
    ]
    if not matches:
        return None
    if len(matches) > 1:
        owners = ", ".join(dataset.owner for dataset in matches)
        raise ValueError(
            f"{len(matches)} datasets are named {name!r} (owned by {owners}). "
            "Pass datasite= to pick one."
        )
    return matches[0]


def wait_for_job(
    sync: Callable[[], Any],
    list_jobs: Callable[[], Iterable[Any]],
    job_name: str,
    user_name: Optional[str],
    status: JobStatusArg,
    where: Optional[JobCheck],
    timeout: float,
    poll_interval: float,
) -> Any:
    """Sync, then look for the job, until ``find_job`` returns it.

    Raises:
        TimeoutError: the job did not match within ``timeout`` seconds.
        ValueError: more than one job matches, or ``status`` is not valid.
        JobEndedError: the job has a final status that is not in ``status``.
    """
    statuses = job_statuses(status)
    wanted = f" to be {' or '.join(sorted(statuses))}" if statuses else ""
    checked = " and to pass the where= check" if where else ""

    def find() -> Optional[Any]:
        sync()
        return find_job(list_jobs(), job_name, user_name, statuses, where)

    return wait_for(
        find,
        f"job {job_name!r}{wanted}{checked}",
        lambda: describe_job(list_jobs(), job_name, user_name, statuses, where),
        timeout,
        poll_interval,
    )


def wait_for_dataset(
    sync: Callable[[], Any],
    list_datasets: Callable[[], Iterable[Any]],
    name: str,
    datasite: Optional[str],
    timeout: float,
    poll_interval: float,
) -> Any:
    """Sync, then look for the dataset, until ``find_dataset`` returns it.

    Raises:
        TimeoutError: no dataset matched within ``timeout`` seconds.
        ValueError: more than one owner has a dataset called ``name``.
    """
    from_owner = f" from {datasite}" if datasite else ""

    def find() -> Optional[Any]:
        sync()
        return find_dataset(list_datasets(), name, datasite)

    return wait_for(
        find,
        f"dataset {name!r}{from_owner}",
        lambda: f"no dataset named {name!r}{from_owner} yet",
        timeout,
        poll_interval,
    )
