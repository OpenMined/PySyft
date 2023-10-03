# stdlib
import argparse
import os
import sys
from typing import Any

# third party
import yaml


# Preserve those beautiful multi-line strings with |
# https://stackoverflow.com/a/33300001
def str_presenter(dumper: Any, data: Any) -> Any:
    if len(data.splitlines()) > 1:  # check for multiline string
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


yaml.add_representer(str, str_presenter)
yaml.representer.SafeRepresenter.add_representer(str, str_presenter)

template_variables = {
    "STACK_API_KEY": "secrets.syft",
    "DEFAULT_ROOT_EMAIL": "secrets.syft",
    "DEFAULT_ROOT_PASSWORD": "secrets.syft",
    "MONGO_PASSWORD": "secrets.db.mongo",
    "MONGO_USERNAME": "secrets.db.mongo",
    "MONGO_INITDB_ROOT_PASSWORD": "secrets.db.mongo",
    "MONGO_INITDB_ROOT_USERNAME": "secrets.db.mongo",
    "MONGO_PORT": "db.mongo.settings",
    "MONGO_HOST": "db.mongo.settings",
    "HOSTNAME": "node.settings",
    "NODE_TYPE": "node.settings",
    "VERSION_HASH": "node.settings",
    "NODE_NAME": "node.settings",
}


def to_lower_camel_case(s: str) -> str:
    words = s.replace("-", "_").split("_")
    return words[0].lower() + "".join(word.capitalize() for word in words[1:])


def remove_yaml(d: Any) -> None:
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


def replace_variables(d: Any) -> None:
    if "name" in d and "value" in d and d["name"] in template_variables:
        variable_name = d["name"]
        path = template_variables[variable_name]
        camel_case_name = to_lower_camel_case(variable_name)
        d["value"] = f"{{{{ .Values.{path}.{camel_case_name} }}}}"

    if "kubernetes.io/ingress.class" in d:
        d["kubernetes.io/ingress.class"] = "{{ .Values.ingress.ingressClass }}"

    if "host" in d:
        d["host"] = "{{ .Values.node.settings.hostname }}"

    if "hosts" in d:
        d["hosts"] = ["{{ .Values.node.settings.hostname }}"]


# parse whole tree
def fix_devspace_yaml(d: Any) -> None:
    if isinstance(d, dict):
        remove_yaml(d)
        replace_variables(d)

        for _, v in d.items():
            fix_devspace_yaml(v)

    elif isinstance(d, list):
        for item in d:
            fix_devspace_yaml(item)


def get_yaml_name(doc: Any) -> Any:
    try:
        return yaml.safe_load(doc).get("metadata", {}).get("name", "")
    except Exception:  # nosec
        return ""


def main() -> None:
    # Argument parsing
    parser = argparse.ArgumentParser(description="Process devspace yaml file.")
    parser.add_argument(
        "file", nargs="?", type=argparse.FileType("r"), default=sys.stdin
    )
    args = parser.parse_args()
    helm_dir = "helm"

    text = args.file.read()

    # input_file = f"{helm_dir}/raw_manifests.yaml"
    # with open(input_file, "w") as f:
    #     f.write(text)

    # Read input from file or stdin
    lines = text.splitlines()

    # Find first line that starts with 'apiVersion' and slice list from that point
    try:
        first_index = next(
            i for i, line in enumerate(lines) if line.strip().startswith("apiVersion")
        )
        input_data = "---\n" + "\n".join(lines[first_index - 1 :])
    except StopIteration:
        print("helm.py error: No line starting with 'apiVersion' found in the input.")
        print("------------------------------")
        print("Got input text:")
        print(text)
        print("------------------------------")
        return

    helm_chart_template_dir = f"{helm_dir}/syft/templates"

    # Split input_data into separate documents
    yaml_docs = input_data.split("---")

    # Sort YAML docs based on metadata name
    yaml_docs.sort(key=get_yaml_name)

    # Join sorted YAML docs
    sorted_input_data = "---".join(yaml_docs)

    # Save sorted YAML docs to file
    input_file = f"{helm_dir}/manifests.yaml"
    with open(input_file, "w") as f:
        f.write(sorted_input_data)

    for doc in yaml_docs:
        lines = doc.strip().split("\n")
        if len(lines) <= 2:
            continue  # skip empty sections

        output_dir = os.path.join(helm_chart_template_dir)

        # Create directories if they don't exist
        os.makedirs(output_dir, exist_ok=True)

        # Parse yaml to find metadata.name
        yaml_content = yaml.safe_load("\n".join(lines))  # exclude source_line
        fix_devspace_yaml(yaml_content)
        name = yaml_content.get("metadata", {}).get("name")
        kind = yaml_content.get("kind", "").lower()
        if name:
            # Create new file with name or append if it already exists
            new_file = os.path.join(output_dir, f"{name}-{kind}.yaml")
            yaml_dump = yaml.dump(yaml_content)
            yaml_dump = (
                yaml_dump.replace("'{{", "{{")
                .replace("}}'", "}}")
                .replace("''{{", "{{")
                .replace("}}''", "}}")
            )

            with open(new_file, "w") as f:
                f.write(yaml_dump)  # add document separator


if __name__ == "__main__":
    main()
