# stdlib
import subprocess

# third party
import yaml


def latest_commit_id() -> str:
    cmd = 'git log --format="%H" -n 1'
    commit_id = subprocess.check_output(cmd, shell=True)
    return commit_id.decode("utf-8").strip()


def update_manifest() -> None:
    """Update manifest_template file with latest commit hash."""

    # Get latest commit id
    commit_id = latest_commit_id()

    # open the manifest file
    with open("hagrid/manifest_template.yml", "r") as stream:
        template_dict = yaml.safe_load(stream)

    # update commit id
    template_dict["hash"] = commit_id

    # save manifest file
    with open("hagrid/manifest_template.yml", "w") as fp:
        yaml.dump(template_dict, fp, sort_keys=False)


if __name__ == "__main__":
    update_manifest()  # Update manifest file
