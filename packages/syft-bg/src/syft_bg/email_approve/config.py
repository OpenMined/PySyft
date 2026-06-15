"""Configuration for email-based approval service."""

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from syft_bg.common.config import get_default_paths, load_yaml, save_yaml


class EmailApproveConfig(BaseModel):
    """Configuration for the email approval service."""

    do_email: Optional[str] = None
    syftbox_root: Optional[Path] = None
    gmail_token_path: Path = Field(
        default_factory=lambda: get_default_paths().gmail_token
    )
    gcp_project_id: Optional[str] = None
    pubsub_topic: Optional[str] = None
    pubsub_subscription: Optional[str] = None
    email_approve_state_path: Path = Field(
        default_factory=lambda: get_default_paths().email_approve_state
    )
    notify_state_path: Path = Field(
        default_factory=lambda: get_default_paths().notify_state
    )

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "EmailApproveConfig":
        """Load config from YAML file."""
        if config_path is None:
            config_path = get_default_paths().config

        if not config_path.exists():
            return cls()

        data = load_yaml(config_path)
        common_cfg = {k: v for k, v in data.items() if not isinstance(v, dict)}
        email_approve_cfg = data.get("email_approve", {})

        do_email = common_cfg.get("do_email")
        syftbox_root = common_cfg.get("syftbox_root")

        gcp_project_id = email_approve_cfg.get("gcp_project_id")
        pubsub_topic = email_approve_cfg.get("pubsub_topic")
        pubsub_subscription = email_approve_cfg.get("pubsub_subscription")

        return cls(
            do_email=do_email,
            syftbox_root=Path(syftbox_root) if syftbox_root else None,
            gcp_project_id=gcp_project_id,
            pubsub_topic=pubsub_topic,
            pubsub_subscription=pubsub_subscription,
        )

    def save_pubsub_config(self, config_path: Optional[Path] = None) -> None:
        """Save Pub/Sub settings back to config.yaml."""
        if config_path is None:
            config_path = get_default_paths().config

        data = load_yaml(config_path)
        if "email_approve" not in data:
            data["email_approve"] = {}

        data["email_approve"]["gcp_project_id"] = self.gcp_project_id
        data["email_approve"]["pubsub_topic"] = self.pubsub_topic
        data["email_approve"]["pubsub_subscription"] = self.pubsub_subscription
        save_yaml(config_path, data)
