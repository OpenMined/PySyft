"""Loads the policies of the api a job matched, for the job runner."""

from syft_rds.apis.policies import ApiPolicyResolver

from syft_bg.approve.api_store import ApiStore


def api_policy_resolver(api_store: ApiStore) -> ApiPolicyResolver:
    """Resolve api names against api_store, checking the policy hashes."""
    return ApiPolicyResolver(api_store.apis_dir)
