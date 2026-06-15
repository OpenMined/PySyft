"""Auto-create Pub/Sub topic and subscription for Gmail push notifications."""

import json
from pathlib import Path

from google.api_core.exceptions import AlreadyExists, NotFound
from google.cloud import pubsub_v1
from google.iam.v1 import iam_policy_pb2
from google.oauth2.credentials import Credentials

GMAIL_PUBLISH_SA = "gmail-api-push@system.gserviceaccount.com"


def get_project_id_from_credentials(credentials_path: Path) -> str:
    """Extract GCP project ID from OAuth credentials.json."""
    data = json.loads(credentials_path.read_text())
    for key in ("installed", "web"):
        if key in data and "project_id" in data[key]:
            return data[key]["project_id"]
    raise ValueError(
        f"Could not find project_id in {credentials_path}. "
        "Ensure credentials.json contains a project_id field."
    )


def ensure_topic(credentials: Credentials, project_id: str, topic_id: str) -> str:
    """Create Pub/Sub topic if it doesn't exist. Returns full topic path."""
    publisher = pubsub_v1.PublisherClient(credentials=credentials)
    topic_path = publisher.topic_path(project_id, topic_id)

    try:
        publisher.create_topic(request={"name": topic_path})
        print(f"[PubSubSetup] Created topic: {topic_path}")
    except AlreadyExists:
        print(f"[PubSubSetup] Topic already exists: {topic_path}")

    return topic_path


def ensure_subscription(
    credentials: Credentials, project_id: str, topic_id: str, subscription_id: str
) -> str:
    """Create Pub/Sub subscription if it doesn't exist. Returns full path."""
    subscriber = pubsub_v1.SubscriberClient(credentials=credentials)
    publisher = pubsub_v1.PublisherClient(credentials=credentials)
    topic_path = publisher.topic_path(project_id, topic_id)
    sub_path = subscriber.subscription_path(project_id, subscription_id)

    try:
        subscriber.create_subscription(request={"name": sub_path, "topic": topic_path})
        print(f"[PubSubSetup] Created subscription: {sub_path}")
    except AlreadyExists:
        print(f"[PubSubSetup] Subscription already exists: {sub_path}")

    return sub_path


def grant_gmail_publish_access(
    credentials: Credentials, project_id: str, topic_id: str
) -> None:
    """Grant Gmail's service account publish access to the topic."""
    publisher = pubsub_v1.PublisherClient(credentials=credentials)
    topic_path = publisher.topic_path(project_id, topic_id)

    try:
        policy = publisher.get_iam_policy(
            request=iam_policy_pb2.GetIamPolicyRequest(resource=topic_path)
        )
    except NotFound:
        print(f"[PubSubSetup] Topic not found: {topic_path}")
        return

    member = f"serviceAccount:{GMAIL_PUBLISH_SA}"
    role = "roles/pubsub.publisher"

    for binding in policy.bindings:
        if binding.role == role and member in binding.members:
            print("[PubSubSetup] Gmail publish access already granted")
            return

    policy.bindings.add(role=role, members=[member])
    publisher.set_iam_policy(
        request=iam_policy_pb2.SetIamPolicyRequest(resource=topic_path, policy=policy)
    )
    print(f"[PubSubSetup] Granted Gmail publish access to {topic_path}")


def _load_project_id_from_config() -> str | None:
    """Load gcp_project_id from config.yaml if available."""
    from syft_bg.common.config import get_default_paths, load_yaml

    config_path = get_default_paths().config
    if not config_path.exists():
        return None
    data = load_yaml(config_path)
    return data.get("email_approve", {}).get("gcp_project_id")


def setup_pubsub(
    credentials: Credentials,
    project_id: str | None = None,
    topic_id: str = "syft-gmail-notifications",
    subscription_id: str = "syft-gmail-sub",
) -> tuple[str, str]:
    """Set up all Pub/Sub resources. Returns (topic_path, subscription_path).

    If project_id is not provided, it is loaded from config.yaml.
    """
    if project_id is None:
        project_id = _load_project_id_from_config()
    if project_id is None:
        raise ValueError(
            "gcp_project_id not found. Run syft_bg.authenticate() first "
            "to extract it from credentials.json."
        )
    topic_path = ensure_topic(credentials, project_id, topic_id)
    grant_gmail_publish_access(credentials, project_id, topic_id)
    sub_path = ensure_subscription(credentials, project_id, topic_id, subscription_id)
    return topic_path, sub_path
