# stdlib
import argparse
import os
import shutil
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

    if "kind" in d and d["kind"] == "Ingress" and "spec" in d:
        d["spec"]["tls"] = [{"hosts": ["{{ .Values.node.settings.hostname }}"]}]
        d["spec"]["rules"][0]["host"] = "{{ .Values.node.settings.hostname }}"


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


def get_yaml_name(doc: dict) -> Any:
    try:
        return doc.get("metadata", {}).get("name", "")
    except Exception:  # nosec
        return ""


def apply_patches(yaml: str, resource_name: str) -> str:
    # apply resource specific patches
    if resource_name.startswith("seaweedfs"):
        yaml = (
            '{{- if ne .Values.node.settings.nodeType "gateway"}}\n'
            + yaml.rstrip()
            + "\n{{ end }}\n"
        )

    # global patches
    yaml = (
        yaml.replace("'{{", "{{")
        .replace("}}'", "}}")
        .replace("''{{", "{{")
        .replace("}}''", "}}")
    )

    return yaml


def main() -> None:
    # Argument parsing
    parser = argparse.ArgumentParser(description="Process devspace yaml file.")
    parser.add_argument(
        "file", nargs="?", type=argparse.FileType("r"), default=sys.stdin
    )
    args = parser.parse_args()
    text = args.file.read()

    file_count = 0
    helm_dir = "helm"
    manifest_file = f"{helm_dir}/manifests.yaml"
    helm_chart_template_dir = f"{helm_dir}/syft/templates"

    # Read input from file or stdin
    lines = text.splitlines()

    # Find first line that starts with 'apiVersion' and slice list from that point
    try:
        first_index = next(
            i for i, line in enumerate(lines) if line.strip().startswith("apiVersion")
        )
        input_data = "\n".join(lines[first_index:])
    except StopIteration:
        print("❌ Error: No line starting with 'apiVersion' found in the input.")
        exit(1)

    # Load the multi-doc yaml file
    try:
        yaml_docs = list(yaml.safe_load_all(input_data))
    except Exception as e:
        print(f"❌ Error while parsing yaml file: {e}")
        exit(1)

    # clear templates dir
    shutil.rmtree(helm_chart_template_dir, ignore_errors=True)

    # Create directories if they don't exist
    os.makedirs(helm_chart_template_dir, exist_ok=True)

    # Cleanup YAML docs
    yaml_docs = [doc for doc in yaml_docs if doc]

    # Sort YAML docs based on metadata name
    yaml_docs.sort(key=get_yaml_name)

    # Save sorted YAML docs to file
    with open(manifest_file, "w") as f:
        yaml.dump_all(yaml_docs, f)

    for doc in yaml_docs:
        fix_devspace_yaml(doc)
        name = doc.get("metadata", {}).get("name")
        kind = doc.get("kind", "").lower()
        if name:
            # Create new file with name or append if it already exists
            new_file = os.path.join(helm_chart_template_dir, f"{name}-{kind}.yaml")
            yaml_dump = yaml.dump(doc)
            yaml_dump = apply_patches(yaml_dump, name)

            with open(new_file, "w") as f:
                f.write(yaml_dump)  # add document separator
                file_count += 1

    if file_count > 0:
        print(f"✅ Done: Generated {file_count} template files")
    else:
        print("❌ Failed: No files were generated. Check input for errors.")
        exit(1)


if __name__ == "__main__":
    main()
