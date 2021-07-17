# stdlib
from argparse import ArgumentParser
import importlib
from pathlib import Path
from typing import List

# third party
from configupdater.configupdater import ConfigUpdater
from pyscaffold import structure
from pyscaffold.actions import Action
from pyscaffold.actions import ActionParams
from pyscaffold.actions import ScaffoldOpts
from pyscaffold.actions import Structure
import pyscaffold.dependencies as deps
from pyscaffold.extensions import Extension
from pyscaffold.extensions import include
from pyscaffold.extensions.no_skeleton import NoSkeleton
from pyscaffold.operations import no_overwrite
from pyscaffold.structure import merge
from pyscaffold.templates import add_pyscaffold
from pyscaffold.templates import get_template
from pyscaffoldext.markdown.extension import Markdown

# syft relative
from . import templates
from .script import generate_package_support


class SyftSupport(Extension):
    """
    Generate syft-support package for a lib.
    """

    def augment_cli(self, parser: ArgumentParser) -> ArgumentParser:
        """Augments the command-line interface parser
        A command line argument ``--FLAG`` where FLAG=``self.name`` is added
        which appends ``self.activate`` to the list of extensions. As help
        text the docstring of the extension class is used.
        In most cases this method does not need to be overwritten.
        Args:
            parser: current parser object
        """
        parser.add_argument(
            self.flag,
            help=self.help_text,
            nargs=0,
            action=include(NoSkeleton(), Markdown(), self),
        )
        return self

    def activate(self, actions: List[Action]) -> List[Action]:
        """Activate extension. See :obj:`pyscaffold.extension.
        Extension.activate`."""
        actions = self.register(
            actions, set_pkg_opts, after="verify_options_consistency"
        )
        actions = self.register(actions, add_files, before="verify_project_dir")
        actions = self.unregister(actions, "init_git")
        return actions


def default_opts(opts: ScaffoldOpts) -> ScaffoldOpts:
    opts["author"] = "OpenMined"
    opts["email"] = "info@openmined.org"
    opts["url"] = "https://github.com/OpenMined/PySyft"
    opts["license"] = "Apache-2.0"

    return opts


def set_pkg_opts(struct: Structure, opts: ScaffoldOpts) -> ActionParams:

    path: Path = opts["project_path"]
    if not str(path).startswith("syft."):
        opts["project_path"] = Path("syft." + str(path))

    if not opts["package"].startswith("syft."):
        opts["package"] = "syft." + opts["package"]

    if not opts["name"].startswith("syft."):
        opts["name"] = "syft." + opts["name"]

    module = opts["name"][5:]
    try:
        importlib.import_module(module)
    except ImportError:
        raise ImportError(f"Make support {module} is installed and can be imported")

    if not opts["requirements"]:
        opts["requirements"] = ["syft", module]

    opts = default_opts(opts)

    return struct, opts


def setup_cfg(opts: ScaffoldOpts) -> ActionParams:
    template = get_template("setup_cfg", relative_to=templates.__name__)
    cfg_str = template.substitute(opts)
    updater = ConfigUpdater()
    updater.read_string(cfg_str)
    requirements = deps.add(deps.RUNTIME, opts.get("requirements", []))
    updater["options"]["install_requires"].set_values(requirements)

    # fill [pyscaffold] section used for later updates
    add_pyscaffold(updater, opts)
    pyscaffold = updater["pyscaffold"]
    pyscaffold["version"].add_after.option("package", opts["package"])

    return str(updater)


def add_files(struct: Structure, opts: ScaffoldOpts) -> ActionParams:
    """Add custom extension files. See :obj:`pyscaffold.actions.Action`"""

    init_py = get_template("init", relative_to=templates.__name__)
    build_proto = get_template("build_proto", relative_to=templates.__name__)
    proto_template = get_template("proto", relative_to=templates.__name__)
    proto = proto_template.substitute(core_package=opts["package"][5:])
    serde = get_template("serde", relative_to=templates.__name__)
    conftest_py = get_template("conftest_py", relative_to=templates.__name__)
    VERSION = get_template("VERSION", relative_to=templates.__name__)
    pyproject_toml = get_template("pyproject_toml", relative_to=templates.__name__)

    module = importlib.import_module(opts["name"][5:])
    package_support = generate_package_support(module)

    files: Structure = {
        "setup.cfg": (setup_cfg, no_overwrite()),
        "pyproject.toml": (pyproject_toml, no_overwrite()),
        "proto": {"sample.proto": (proto, no_overwrite())},
        "src": {
            opts["package"]: {
                "__init__.py": (init_py, no_overwrite()),
                "package-support.json": (package_support, no_overwrite()),
                "proto": {},
                "serde": {
                    "sample.py": (serde, no_overwrite()),
                },
                "VERSION": (VERSION, no_overwrite()),
            },
        },
        "tests": {
            "conftest.py": (conftest_py, no_overwrite()),
        },
        "scripts": {
            "build_proto.sh": (build_proto, no_overwrite()),
        },
    }

    # reject files
    struct = structure.reject(struct, Path("docs"))
    struct = structure.reject(struct, Path("setup.py"))
    struct = structure.reject(struct, Path(".readthedocs.yml"))

    return merge(struct, files), opts
