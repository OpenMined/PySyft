"""Configuration for the Remote Data Science product.

``SyftRDSClientConfig`` COMPOSES the three sub-configs the RDS layer owns:

* ``sync``    – the (domain-free) ``SyftboxManagerConfig`` sync engine config,
* ``job``     – the ``SyftJobConfig`` for the RDS-owned ``JobClient``/``SyftJobRunner``,
* ``dataset`` – the ``SyftBoxConfig`` for the RDS-owned ``SyftDatasetManager``.

All three are derived from the SAME primitives (email, syftbox folder, role)
so their paths always line up. The RDS layer is also the place where the
dataset collection sync specs are *supplied* into the generic sync core; the
core itself stays domain-free.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, model_validator

from syft_client.sync.syftbox_manager import (
    SyftboxManagerConfig,
)
from syft_client.sync.sync.collection_spec import CollectionSyncSpec
from syft_job import SyftJobConfig
from syft_datasets.config import SyftBoxConfig
from syft_datasets.dataset_manager import (
    DATASET_COLLECTION_PREFIX,
    PRIVATE_DATASET_COLLECTION_PREFIX,
)

# The RDS layer owns the local subpaths; the on-wire prefixes come from
# syft_datasets (imported above), mirrored in syft_client for login-time cleanup.
COLLECTION_SUBPATH = Path("public/syft_datasets")
PRIVATE_COLLECTION_SUBPATH = Path("private/syft_datasets")

# The RDS layer OWNS the dataset collection specs (syft-client core stays domain-free).
# Two specs, distinguished purely by the two generic behavioral flags:
#   * public (mock)  – mirror + shareable  → peers' watchers pull it.
#   * private (real) – restore-only + owner-only → the owner restores it for itself;
#                      peer-facing watchers skip it; it is never shared.
DATASET_COLLECTION_SPECS = [
    CollectionSyncSpec.public(DATASET_COLLECTION_PREFIX, COLLECTION_SUBPATH),
    CollectionSyncSpec.private(
        PRIVATE_DATASET_COLLECTION_PREFIX, PRIVATE_COLLECTION_SUBPATH
    ),
]


class SyftRDSClientConfig(BaseModel):
    sync: SyftboxManagerConfig
    job: SyftJobConfig
    dataset: SyftBoxConfig

    @model_validator(mode="after")
    def _assert_sub_configs_aligned(self) -> "SyftRDSClientConfig":
        """Enforce the invariant the ``_compose`` docstring promises: all three
        sub-configs are derived from the SAME primitives, so their paths line up.

        Guards against a hand-built config (or a future factory) drifting the
        email / folder / role across the sync engine, job client, and dataset
        manager, which would silently point them at mismatched datasite paths.
        """
        emails = {
            "sync": self.sync.email,
            "job": self.job.current_user_email,
            "dataset": self.dataset.email,
        }
        if len(set(emails.values())) > 1:
            raise ValueError(f"sub-config emails must all match, got {emails}")

        folders = {
            "sync": self.sync.syftbox_folder,
            "job": self.job.syftbox_folder,
            "dataset": self.dataset.syftbox_folder,
        }
        if len(set(folders.values())) > 1:
            raise ValueError(
                f"sub-config syftbox_folders must all match, got {folders}"
            )

        if self.sync.has_do_role != self.job.has_do_role:
            raise ValueError(
                "sync.has_do_role and job.has_do_role must match, got "
                f"{self.sync.has_do_role} vs {self.job.has_do_role}"
            )
        return self

    # ------------------------------------------------------------------ #
    # private helper: build all three sub-configs from shared primitives
    # ------------------------------------------------------------------ #
    @staticmethod
    def _compose(sync: SyftboxManagerConfig) -> "SyftRDSClientConfig":
        """Build the composed config using ``sync`` as the single source of truth.

        ``sync`` is built by the caller (it differs per environment); ``job`` and
        ``dataset`` are derived here from the sync engine's OWN email/folder/role
        rather than from separately-passed primitives, so the sub-configs cannot
        drift apart. The ``_assert_sub_configs_aligned`` validator enforces this.
        """
        job = SyftJobConfig(
            syftbox_folder=sync.syftbox_folder,
            current_user_email=sync.email,
            has_do_role=sync.has_do_role,
        )
        dataset = SyftBoxConfig(syftbox_folder=sync.syftbox_folder, email=sync.email)
        return SyftRDSClientConfig(sync=sync, job=job, dataset=dataset)

    @classmethod
    def for_jupyter(
        cls,
        email,
        has_do_role=False,
        has_ds_role=False,
        token_path=None,
        **kw,
    ) -> "SyftRDSClientConfig":
        sync = SyftboxManagerConfig.for_jupyter(
            email=email,
            has_do_role=has_do_role,
            has_ds_role=has_ds_role,
            token_path=token_path,
            collection_specs=DATASET_COLLECTION_SPECS,
            **kw,
        )
        return cls._compose(sync)

    @classmethod
    def for_colab(
        cls,
        email,
        has_do_role=False,
        has_ds_role=False,
        **kw,
    ) -> "SyftRDSClientConfig":
        sync = SyftboxManagerConfig.for_colab(
            email=email,
            has_do_role=has_do_role,
            has_ds_role=has_ds_role,
            collection_specs=DATASET_COLLECTION_SPECS,
            **kw,
        )
        return cls._compose(sync)

    @classmethod
    def _base_config_for_testing(
        cls,
        email=None,
        syftbox_folder=None,
        has_do_role=False,
        has_ds_role=False,
        **kw,
    ) -> "SyftRDSClientConfig":
        """Build a composed config over ``SyftboxManagerConfig._base_config_for_testing``.

        The sync sub-config is the base testing config (in-/out-of-memory caches,
        mock connections wired in later); ``job`` and ``dataset`` are derived from
        the SAME email + folder that the sync config resolved, so paths align.
        """
        sync = SyftboxManagerConfig._base_config_for_testing(
            email=email,
            syftbox_folder=syftbox_folder,
            has_do_role=has_do_role,
            has_ds_role=has_ds_role,
            collection_specs=DATASET_COLLECTION_SPECS,
            **kw,
        )
        # _base_config_for_testing may have generated a random email/folder;
        # _compose reads them back off the resolved sync config.
        return cls._compose(sync)

    @classmethod
    def for_google_drive_testing_connection(
        cls,
        email,
        token_path,
        syftbox_folder=None,
        has_do_role=False,
        has_ds_role=False,
        **kw,
    ) -> "SyftRDSClientConfig":
        """Build a composed config over ``SyftboxManagerConfig.for_google_drive_testing_connection``.

        Same shape as :meth:`_base_config_for_testing`, but the sync sub-config is
        wired to a REAL Google Drive connection (via ``token_path``) instead of the
        in-memory mock — for integration tests that exercise the actual transport.
        ``job`` and ``dataset`` are derived from the SAME email + folder the sync
        config resolved, so paths align.
        """
        sync = SyftboxManagerConfig.for_google_drive_testing_connection(
            email=email,
            token_path=token_path,
            syftbox_folder=syftbox_folder,
            has_do_role=has_do_role,
            has_ds_role=has_ds_role,
            collection_specs=DATASET_COLLECTION_SPECS,
            **kw,
        )
        return cls._compose(sync)
