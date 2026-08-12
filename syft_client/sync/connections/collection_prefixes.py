"""Wire-level collection folder-name prefixes."""

# Mirrors syft_datasets.dataset_manager. Duplicated rather than imported because
# delete_unversioned_state needs these at login time, before an RDS client exists,
# and the sync core must not import the domain.

# Kept in sync by test_collection_prefixes_match_syft_datasets in
# packages/syft-rds/tests.

DATASET_COLLECTION_PREFIX = "syft_datasetcollection"
PRIVATE_DATASET_COLLECTION_PREFIX = "syft_privatecollection"
