# stdlib
from platform import python_version_tuple

python_version = python_version_tuple()

if (int(python_version[0]), int(python_version[1])) >= (3, 8):
    # stdlib
    from importlib.metadata import version

    __version__ = version("mongomock")
else:
    # third party
    import pkg_resources

    __version__ = pkg_resources.get_distribution("mongomock").version
