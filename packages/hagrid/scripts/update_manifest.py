# stdlib
import os
import subprocess
import sys
from typing import Optional

# third party
import yaml


def latest_commit_id() -> str:
    cmd = 'git log --format="%H" -n 1'
    commit_id = subprocess.check_output(cmd, shell=True)
    return commit_id.decode("utf-8").strip()


def update_manifest(docker_tag: Optional[str]) -> None:
    """Update manifest_template file with latest commit hash."""

    # Get latest commit id
    commit_id = latest_commit_id()

    template_dir = os.path.abspath(os.path.join(os.path.realpath(__file__), "../../"))
    template_filepath = os.path.join(template_dir, "hagrid/manifest_template.yml")

    # open the manifest file
    with open(template_filepath, "r") as stream:
        template_dict = yaml.safe_load(stream)

    # update commit id
    template_dict["hash"] = commit_id

    # update docker tag if available
    if docker_tag:
        template_dict["dockerTag"] = docker_tag

    # save manifest file
    with open(template_filepath, "w") as fp:
        yaml.dump(template_dict, fp, sort_keys=False)


if __name__ == "__main__":

    docker_tag = None

    if len(sys.argv) > 1:
        docker_tag = sys.argv[1]

    update_manifest(docker_tag)  # Update manifest file
