from importlib.metadata import version

# Derived from the installed distribution metadata (pyproject.toml's version)
# so it cannot drift from the released package version.
__version__ = version("syft-job")
