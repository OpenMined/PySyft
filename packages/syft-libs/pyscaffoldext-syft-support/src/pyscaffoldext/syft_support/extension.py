import importlib
from pathlib import Path
from typing import List

from pyscaffold.actions import Action, ActionParams, ScaffoldOpts, Structure
from pyscaffold.extensions import Extension, include
from pyscaffold.extensions.no_skeleton import NoSkeleton
from pyscaffold.extensions.pre_commit import PreCommit
from pyscaffold.operations import no_overwrite
from pyscaffold.structure import merge
from pyscaffold.templates import get_template

from . import templates
from .script import generate_package_support


class SyftSupport(Extension):
    """
    This class serves as the skeleton for your new PyScaffold Extension. Refer
    to the official documentation to discover how to implement a PyScaffold
    extension - https://pyscaffold.org/en/latest/extensions.html
    """

    def augment_cli(self, parser):
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
            action=include(NoSkeleton(), PreCommit(), self),
        )
        return self

    def activate(self, actions: List[Action]) -> List[Action]:
        """Activate extension. See :obj:`pyscaffold.extension.
        Extension.activate`."""
        actions = self.register(
            actions, change_package_name, after="verify_options_consistency"
        )
        actions = self.register(actions, add_files)
        actions = self.unregister(actions, "init_git")
        return actions


def change_package_name(struct: Structure, opts: ScaffoldOpts) -> ActionParams:

    path: Path = opts["project_path"]
    if not str(path).startswith("syft-"):
        opts["project_path"] = Path("syft-" + str(path))

    if not opts["package"].startswith("syft-"):
        opts["package"] = "syft_" + opts["package"]

    if not opts["name"].startswith("syft-"):
        opts["name"] = "syft-" + opts["name"]

    module = opts["name"][5:]
    try:
        importlib.import_module(module)
    except ImportError:
        raise ImportError(
            "Make support {} is installed and can be imported".format(module)
        )

    return struct, opts


def add_files(struct: Structure, opts: ScaffoldOpts) -> ActionParams:
    """Add custom extension files. See :obj:`pyscaffold.actions.Action`"""

    init_template = get_template("init", relative_to=templates.__name__)
    build_proto_template = get_template("build_proto", relative_to=templates.__name__)
    proto_template = get_template("sample_proto", relative_to=templates.__name__)
    proto_sample = proto_template.substitute(core_package=opts["package"][5:])
    serde_template = get_template("sample_serde", relative_to=templates.__name__)

    module = importlib.import_module(opts["name"][5:])
    package_support = generate_package_support(module)

    # package_support_template = get_template("sample_package_support", relative_to=templates.__name__)

    files: Structure = {
        "proto": {"sample.proto": (proto_sample, no_overwrite())},
        "src": {
            opts["package"]: {
                "__init__.py": (init_template, no_overwrite()),
                "package-suport.json": (package_support, no_overwrite()),
                "proto": {},
                "serde": {
                    "sample.py": (serde_template, no_overwrite()),
                },
            },
        },
        "build_proto.sh": (build_proto_template, no_overwrite()),
    }

    return merge(struct, files), opts
