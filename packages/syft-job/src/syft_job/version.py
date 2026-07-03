from importlib.metadata import version

# Distribution name, computed from the import package name ("syft_job" -> "syft-job").
PACKAGE_NAME = __package__.replace("_", "-")

# Derived from the installed distribution metadata (pyproject.toml's version)
# so it cannot drift from the released package version.
__version__ = version(PACKAGE_NAME)
