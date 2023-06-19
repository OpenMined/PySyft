# stdlib
import argparse
import os
import sys
from typing import Any

# third party
import yaml


def fix_devspace_yaml(d: Any) -> None:
    if isinstance(d, dict):
        if "namespace" in d:
            del d["namespace"]
        if (
            "kind" in d
            and d["kind"] == "Deployment"
            and "spec" in d
            and "volumeClaimTemplates" in d["spec"]
            and d["spec"]["volumeClaimTemplates"] is None
        ):
            del d["spec"]["volumeClaimTemplates"]

        for _k, v in d.items():
            fix_devspace_yaml(v)
    elif isinstance(d, list):
        for item in d:
            fix_devspace_yaml(item)


def main() -> None:
    # Argument parsing
    parser = argparse.ArgumentParser(description="Process devspace yaml file.")
    parser.add_argument(
        "file", nargs="?", type=argparse.FileType("r"), default=sys.stdin
    )
    args = parser.parse_args()

    # Read input from file or stdin
    input_data = args.file.read()

    # Split input_data into separate documents
    yaml_docs = input_data.split("---")
    input_file = "helm/manifests.yaml"
    with open(input_file, "w") as f:
        f.write(input_data)

    parent_file = None
    for doc in yaml_docs:
        lines = doc.strip().split("\n")
        if len(lines) <= 2:
            continue  # skip empty sections

        source_line = lines[0]
        if source_line.startswith("# Source:"):
            path = source_line.split(": ", 1)[1]  # get the path
            output_dir = os.path.join(
                "helm", os.path.dirname(path)
            )  # add 'helm' subdirectory
            output_file = os.path.basename(path)

            # Create directories if they don't exist
            os.makedirs(output_dir, exist_ok=True)

            # Parse yaml to find metadata.name
            yaml_content = yaml.safe_load("\n".join(lines[1:]))  # exclude source_line
            fix_devspace_yaml(yaml_content)
            name = yaml_content.get("metadata", {}).get("name")
            if name:
                # Create new file with name or append if it already exists
                new_file = os.path.join(output_dir, f"{name}.yaml")
                if os.path.exists(new_file):
                    mode = "a"  # append if file exists
                else:
                    mode = "w"  # write if new file

                with open(new_file, mode) as f:
                    f.write("---\n" + yaml.dump(yaml_content))  # add document separator

                # Append to parent file a reference to the new file
                doc_to_append = (
                    f"\n---\n# Source: {new_file}\n{yaml.dump(yaml_content)}"
                )
            else:
                # If no name, then it's the parent file
                parent_file = os.path.join(output_dir, output_file)
                doc_to_append = doc

            # Append to the file, don't overwrite
            if parent_file:
                with open(parent_file, "a") as f:
                    f.write("---\n" + doc_to_append)  # add document separator


if __name__ == "__main__":
    main()
