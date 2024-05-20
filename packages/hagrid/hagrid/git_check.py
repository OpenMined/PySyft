# relative
from .deps import DependencyGridGit
from .deps import check_deps


def verify_git_installation() -> None:
    dep = DependencyGridGit(name="git", output_in_text=True)
    deps = {}
    deps["git"] = dep
    check_deps(of="Git", deps=deps, display=False, output_in_text=True)  # type: ignore
    if dep.issues:
        exit(1)


verify_git_installation()
