"""Loads the api a request calls, for the job runner and the job client."""

from syft_rds.apis.policies import ApiResolver

from syft_bg.approve.api_store import ApiStore


def api_resolver(api_store: ApiStore) -> ApiResolver:
    """Resolve api names against api_store, checking every file's hash."""
    return ApiResolver(api_store.apis_dir)
