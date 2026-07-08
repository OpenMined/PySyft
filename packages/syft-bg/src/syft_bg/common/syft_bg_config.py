"""Top-level SyftBg configuration model (mirrors config.yaml)."""

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from syft_bg.approve.config import AutoApproveConfig
from syft_bg.common.config import get_default_paths
from syft_bg.common.locking import file_lock
from syft_bg.email_approve.config import EmailApproveConfig
from syft_bg.notify.config import NotifyConfig
from syft_bg.sync.config import SyncConfig


class SyftBgConfig(BaseModel):
    """Top-level syft-bg configuration, matching the config.yaml structure."""

    # Preserve top-level keys we don't model (e.g. written by another
    # syft-bg version) instead of silently dropping them on save().
    model_config = ConfigDict(extra="allow")

    do_email: str | None = None
    syftbox_root: str | Path | None = None
    credentials_path: Path = Field(
        default_factory=lambda: get_default_paths().credentials
    )
    gmail_token_path: Path = Field(
        default_factory=lambda: get_default_paths().gmail_token
    )
    drive_token_path: Path | None = Field(
        default_factory=lambda: get_default_paths().drive_token
    )
    notify: NotifyConfig = Field(default_factory=NotifyConfig)
    approve: AutoApproveConfig = Field(default_factory=AutoApproveConfig)
    email_approve: EmailApproveConfig = Field(default_factory=EmailApproveConfig)
    sync: SyncConfig = Field(default_factory=SyncConfig)

    @model_validator(mode="before")
    @classmethod
    def _set_default_syftbox_root(cls, data: dict) -> dict:
        """Default syftbox_root to ~/SyftBox_{do_email} when not set."""
        if not isinstance(data, dict):
            return data
        if not data.get("syftbox_root") and data.get("do_email"):
            from syft_client.sync.syftbox_manager import (
                get_jupyter_default_syftbox_folder,
            )

            data = dict(data)
            data["syftbox_root"] = str(
                get_jupyter_default_syftbox_folder(data["do_email"])
            )
        return data

    def _merge_common_into_services(self) -> None:
        """Propagate top-level fields into service configs where not already set."""
        for service_config in (
            self.notify,
            self.approve,
            self.email_approve,
            self.sync,
        ):
            if hasattr(service_config, "do_email") and service_config.do_email is None:
                service_config.do_email = self.do_email
            if (
                hasattr(service_config, "syftbox_root")
                and service_config.syftbox_root is None
            ):
                if self.syftbox_root is not None:
                    service_config.syftbox_root = Path(self.syftbox_root)
            # These fields default to a concrete path, not None, so use the
            # default itself (not "is None") to detect an unset field.
            default_paths = get_default_paths()
            path_field_defaults = {
                "drive_token_path": default_paths.drive_token,
                "gmail_token_path": default_paths.gmail_token,
            }
            for path_field, own_default in path_field_defaults.items():
                if hasattr(service_config, path_field):
                    parent_val = getattr(self, path_field, None)
                    current_val = getattr(service_config, path_field, None)
                    if parent_val is not None and current_val == own_default:
                        setattr(service_config, path_field, parent_val)

    def set_service_config(self, name: str, config: dict) -> None:
        subconfig = getattr(self, name)
        for key, value in config.items():
            if key not in subconfig.model_fields:
                raise ValueError(f"Unknown config key: {key}")
            field_type = subconfig.model_fields[key].annotation
            if (
                isinstance(value, dict)
                and isinstance(field_type, type)
                and issubclass(field_type, BaseModel)
            ):
                value = field_type.model_validate(value)
            setattr(subconfig, key, value)

    def _repr_html_(self) -> str:
        """
        Display config as pretty YAML in Jupyter/HTML.
        """
        import yaml

        data = self.model_dump(mode="json")
        yaml_str = yaml.safe_dump(data, sort_keys=False, default_flow_style=False)
        return f"""<b>SyftBgConfig:</b> <br>
location: {get_default_paths().config} <br>
.save() to store
<pre>{yaml_str}</pre>"""

    @classmethod
    def from_path(
        cls, config_path: Path | None = None, *, merge: bool = True
    ) -> "SyftBgConfig":
        """Load from a YAML config file. Pass merge=False before saving back
        (e.g. edit()) to avoid baking inherited fields into sub-configs."""
        if config_path is None:
            config_path = get_default_paths().config

        with open(config_path) as f:
            data = yaml.safe_load(f) or {}

        config = cls.model_validate(data)
        if merge:
            config._merge_common_into_services()
        return config

    def save(self, config_path: Path | None = None) -> None:
        """Write to a YAML config file."""
        if config_path is None:
            config_path = get_default_paths().config

        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, "w") as f:
            yaml.dump(
                self.model_dump(mode="json"),
                f,
                default_flow_style=False,
                sort_keys=False,
            )

    @classmethod
    def load(
        cls, config_path: Path | None = None, *, merge: bool = True
    ) -> "SyftBgConfig":
        """Load config from path, returning defaults if the file doesn't exist."""
        try:
            return cls.from_path(config_path, merge=merge)
        except FileNotFoundError:
            return cls()

    @classmethod
    @contextmanager
    def edit(cls, config_path: Path | None = None) -> Iterator["SyftBgConfig"]:
        """Locked load-mutate-save cycle. Skips save() if the block raises."""
        if config_path is None:
            config_path = get_default_paths().config

        lock_path = config_path.with_suffix(".lock")
        with file_lock(lock_path):
            config = cls.load(config_path, merge=False)
            yield config
            config.save(config_path)
