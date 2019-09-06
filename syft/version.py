__version__ = "0.1.26a1"

import importlib
import types
import codecs
import distutils
import locale
import os
import platform
import struct
import subprocess
import sys
import warnings

VERSIONS = {
    "flask": "1.0.2",
    "flask_socketio": "3.3.2",
    "lz4": "2.1.6",
    "numpy": "1.14.0",
    "sklearn": "0.21.0",
    "tblib": "1.4.0",
    "torch": "1.1",
    "websocket": "0.56.0",
    "websockets": "7.0",
}

version_message = (
    "PySyft requires version '{minimum_version}' or newer of '{name}' "
    "(version '{actual_version}' currently installed)."
)

message = "Missing optional dependency '{name}'. {extra} " "Use pip or conda to install {name}."


def _get_version(module: types.ModuleType) -> str:
    version = getattr(module, "__version__", None)
    if version is None:
        # xlrd uses a capitalized attribute name
        version = getattr(module, "__VERSION__", None)

    if version is None:
        raise ImportError("Can't determine version for {}".format(module.__name__))
    return version


def import_optional_dependency(
    name: str, extra: str = "", raise_on_missing: bool = True, on_version: str = "raise"
):
    """
    Import an optional dependency.

    By default, if a dependency is missing an ImportError with a nice
    message will be raised. If a dependency is present, but too old,
    we raise.

    Parameters
    ----------
    name : str
        The module name. This should be top-level only, so that the
        version may be checked.
    extra : str
        Additional text to include in the ImportError message.
    raise_on_missing : bool, default True
        Whether to raise if the optional dependency is not found.
        When False and the module is not present, None is returned.
    on_version : str {'raise', 'warn'}
        What to do when a dependency's version is too old.

        * raise : Raise an ImportError
        * warn : Warn that the version is too old. Returns None
        * ignore: Return the module, even if the version is too old.
          It's expected that users validate the version locally when
          using ``on_version="ignore"`` (see. ``io/html.py``)

    Returns
    -------
    maybe_module : Optional[ModuleType]
        The imported module, when found and the version is correct.
        None is returned when the package is not found and `raise_on_missing`
        is False, or when the package's version is too old and `on_version`
        is ``'warn'``.
    """
    try:
        module = importlib.import_module(name)
    except ImportError:
        if raise_on_missing:
            raise ImportError(message.format(name=name, extra=extra)) from None
        else:
            return None

    minimum_version = VERSIONS.get(name)
    if minimum_version:
        version = _get_version(module)
        if distutils.version.LooseVersion(version) < minimum_version:
            assert on_version in {"warn", "raise", "ignore"}
            msg = version_message.format(
                minimum_version=minimum_version, name=name, actual_version=version
            )
            if on_version == "warn":
                warnings.warn(msg, UserWarning)
                return None
            elif on_version == "raise":
                raise ImportError(msg)

    return module


def get_sys_info():
    "Returns system information as a dict"

    blob = []

    # get full commit hash
    commit = None
    if os.path.isdir(".git") and os.path.isdir("syft"):
        try:
            pipe = subprocess.Popen(
                'git log --format="%H" -n 1'.split(" "),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            so, serr = pipe.communicate()
        except (OSError, ValueError):
            pass
        else:
            if pipe.returncode == 0:
                commit = so
                try:
                    commit = so.decode("utf-8")
                except ValueError:
                    pass
                commit = commit.strip().strip('"')

    blob.append(("commit", commit))

    try:
        (sysname, nodename, release, version, machine, processor) = platform.uname()
        blob.extend(
            [
                ("python", ".".join(map(str, sys.version_info))),
                ("python-bits", struct.calcsize("P") * 8),
                ("OS", "{sysname}".format(sysname=sysname)),
                ("OS-release", "{release}".format(release=release)),
                # ("Version", "{version}".format(version=version)),
                ("machine", "{machine}".format(machine=machine)),
                ("processor", "{processor}".format(processor=processor)),
                ("byteorder", "{byteorder}".format(byteorder=sys.byteorder)),
                ("LC_ALL", "{lc}".format(lc=os.environ.get("LC_ALL", "None"))),
                ("LANG", "{lang}".format(lang=os.environ.get("LANG", "None"))),
                ("LOCALE", ".".join(map(str, locale.getlocale()))),
            ]
        )
    except (KeyError, ValueError):
        pass

    return blob


def show_versions(as_json=False):
    sys_info = get_sys_info()
    deps = [
        "syft",
        "flask",
        "flask_socketio",
        "lz4",
        "numpy",
        "sklearn",
        "tblib",
        "torch",
        "websocket",
        "websockets",
    ]

    deps_blob = []
    for modname in deps:
        mod = import_optional_dependency(modname, raise_on_missing=False, on_version="ignore")
        if mod:
            ver = _get_version(mod)
        else:
            ver = None
        deps_blob.append((modname, ver))

    if as_json:
        try:
            import json
        except ImportError:
            print("JSON not installed")

        j = dict(system=dict(sys_info), dependencies=dict(deps_blob))

        if as_json is True:
            print(j)
        else:
            with codecs.open(as_json, "wb", encoding="utf8") as f:
                json.dump(j, f, indent=2)

    else:
        maxlen = max(len(x) for x in deps)
        tpl = "{{k:<{maxlen}}}: {{stat}}".format(maxlen=maxlen)
        print("\nINSTALLED VERSIONS")
        print("------------------")
        for k, stat in sys_info:
            print(tpl.format(k=k, stat=stat))
        print("")
        for k, stat in deps_blob:
            print(tpl.format(k=k, stat=stat))
