from importlib.metadata import version

# Distribution name from pyproject ([project].name). The import package is
# "syft_datasets" but the distribution is "syft-dataset" (singular), so this is
# hardcoded rather than derived from __package__.
PACKAGE_NAME = "syft-dataset"

# Derived from the installed distribution metadata (pyproject.toml's version)
# so it cannot drift from the released package version.
__version__ = version(PACKAGE_NAME)
