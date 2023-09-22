# stdlib
import json
import os
import subprocess
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple

# third party
import git
import requests

DEV_MODE = False
KEY = None
JSON_DATA = os.path.dirname(__file__) + "/staging.json"


def run_hagrid(node: Dict) -> int:
    name = node["name"]
    node_type = node["node_type"]
    ip = node["ip"]
    branch = node["branch"]
    cmd = (
        f"hagrid launch {name} {node_type} to {ip} --username=azureuser --auth-type=key "
        f"--key-path={KEY} --repo=OpenMined/PySyft --branch={branch} --verbose"
    )
    watch_shell(cmd)


def watch_shell(command: str) -> None:
    process = subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )  # nosec
    while True:
        output = process.stdout.readline().decode()
        if output == "" and process.poll() is not None:
            break
        if output:
            print(output.strip())
    rc = process.poll()
    return rc


def shell(command: str) -> str:
    try:
        output = subprocess.check_output(  # nosec
            command, shell=True, stderr=subprocess.STDOUT
        )
    except Exception:
        output = b""
    return output.decode("utf-8").strip()


def metadata_url(node: Dict) -> str:
    ip = node["ip"]
    endpoint = node["metadata_endpoint"]
    return f"http://{ip}{endpoint}"


def check_metadata(node: Dict) -> Optional[Dict]:
    try:
        res = requests.get(metadata_url(node))
        if res.status_code != 200:
            print(f"Got status_code: {res.status_code}")
        metadata = res.json()
        name = node["name"]
        print(f"{name} syft_version: ", metadata["syft_version"])
        return metadata
    except Exception as e:
        print(f"Failed to get metadata. {e}")
    return None


def process_node(node: Dict[str, Any]) -> Tuple[bool, str]:
    repo_hash = get_repo_checkout(node)
    metadata = check_metadata(node)
    hash_string = check_remote_hash(node)
    redeploy = False
    if metadata is None or hash_string is None:
        print(f"redeploy because metadata: {metadata} and remote hash: {hash_string}")
        redeploy = True

    if hash_string is not None and repo_hash != hash_string:
        print("repo_hash", len(repo_hash), type(repo_hash))
        print("hash_string", len(hash_string), type(hash_string))
        print(
            f"redeploy because repo_hash: {repo_hash} != remote hash_string: {hash_string}"
        )
        redeploy = True

    if redeploy:
        print("🔧 Redeploying with HAGrid")
        run_hagrid(node)

    hash_string = check_remote_hash(node)
    if hash_string is None:
        print(f"Cant get hash: {hash_string}")

    if hash_string is not None and hash_string != repo_hash:
        print(
            f"Hash doesnt match repo_hash: {repo_hash} != remote hash_string {hash_string}"
        )

    metadata = check_metadata(node)
    if metadata is None:
        print(f"Cant get metadata: {metadata}")

    if metadata and hash_string == repo_hash:
        return True, repo_hash
    return False, repo_hash


def get_repo_checkout(node: Dict) -> str:
    try:
        branch = node["branch"]
        repo_path = f"/tmp/{branch}/PySyft"
        if not os.path.exists(repo_path):
            os.makedirs(repo_path, exist_ok=True)
            repo = git.Repo.clone_from(
                "https://github.com/OpenMined/pysyft",
                repo_path,
                single_branch=True,
                b=branch,
            )
        else:
            repo = git.Repo(repo_path)
        if repo.is_dirty():
            repo.git.reset("--hard")
        repo.git.checkout(branch)
        repo.remotes.origin.pull()
        sha = repo.head.commit.hexsha
        return sha
    except Exception as e:
        print(f"Failed to get branch HEAD commit hash. {e}")
        raise e


def run_remote_shell(node: Dict, cmd: str) -> Optional[str]:
    try:
        ip = node["ip"]
        ssh_cmd = (
            f"ssh -o LogLevel=ERROR -o StrictHostKeyChecking=no -i {KEY} azureuser@{ip}"
        )
        shell_cmd = f'{ssh_cmd} "{cmd}"'
        print("Running:", shell_cmd)
        return shell(shell_cmd)
    except Exception:
        print("Failed to run ssh command: {}")
    return None


def check_remote_hash(node: Dict) -> Optional[str]:
    cmd = "sudo runuser -l om -c 'cd /home/om/PySyft && git rev-parse HEAD'"
    return run_remote_shell(node, cmd)


def check_staging() -> None:
    nodes = load_staging_data(JSON_DATA)
    for name, node in nodes.items():
        print(f"Processing {name}")
        good = False
        try:
            good, updated_hash = process_node(node=node)
            node["commit_hash"] = updated_hash
            nodes[name] = node
            save_staging_data(JSON_DATA, nodes)
        except Exception as e:
            print(f"Failed to process node: {name}. {e}")
        emoji = "✅" if good else "❌"
        print(f"{emoji} Node {name}")


def load_staging_data(path: str) -> Dict[str, Dict]:
    with open(path) as f:
        return json.loads(f.read())


def save_staging_data(path: str, data: Dict[str, Dict]) -> None:
    print("Saving changes to file", path)
    with open(path, "w") as f:
        f.write(f"{json.dumps(data)}")


if __name__ == "__main__":
    # stdlib
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", action="store_true", help="Dev Mode")
    parser.add_argument("--private-key", help="Dev Mode")

    args = parser.parse_args()
    if args.dev:
        DEV_MODE = True
    if args.private_key:
        path = os.path.expanduser(args.private_key)
        if os.path.exists(path):
            KEY = path
    if KEY is None:
        raise Exception("--private-key required")
    print("DEV MODE", DEV_MODE)

    check_staging()
