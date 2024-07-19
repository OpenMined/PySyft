# stdlib
from collections import defaultdict
import json
import os
from pathlib import Path
from typing import Any

# syft absolute
import syft as sy

# relative
from .decorators import deprecated

RELATIVE_PATH_TO_FRONTEND = "/../../../../grid/frontend/"
SCHEMA_FOLDER = "schema"

GUEST_COMMANDS = """
<li><span class='syft-code-block'>&lt;your_client&gt;.datasets</span> - list datasets</li>
<li><span class='syft-code-block'>&lt;your_client&gt;.code</span> - list code</li>
<li><span class='syft-code-block'>&lt;your_client&gt;.login</span> - list projects</li>
"""

DS_COMMANDS = """
<li><span class='syft-code-block'>&lt;your_client&gt;.datasets</span> - list datasets</li>
<li><span class='syft-code-block'>&lt;your_client&gt;.code</span> - list code</li>
<li><span class='syft-code-block'>&lt;your_client&gt;.projects</span> - list projects</li>
"""

DO_COMMANDS = """
<li><span class='syft-code-block'>&lt;your_client&gt;.projects</span> - list projects</li>
<li><span class='syft-code-block'>&lt;your_client&gt;.requests</span> - list requests</li>
<li><span class='syft-code-block'>&lt;your_client&gt;.users</span> - list users</li>
"""

DEFAULT_WELCOME_MSG = """
        <style>
            $FONT_CSS

            .syft-container {
                padding: 5px;
                font-family: 'Open Sans';
            }
            .syft-alert-info {
                color: #1F567A;
                background-color: #C2DEF0;
                border-radius: 4px;
                padding: 5px;
                padding: 13px 10px
            }
            .syft-code-block {
                background-color: #f7f7f7;
                border: 1px solid #cfcfcf;
                padding: 0px 2px;
            }
            .syft-space {
                margin-top: 1em;
            }
        </style>
        <div class="syft-client syft-container">
            <img src="$server_symbol" alt="Logo"
            style="width:48px;height:48px;padding:3px;">
            <h2>Welcome to $datasite_name</h2>
            <div class="syft-space">
            <strong>URL:</strong> $server_url <br />
            <strong>Server Description:</strong> $description <br />
            <strong>Server Type:</strong> $server_type <br />
            <strong>Server Side Type:</strong>$server_side_type<br />
            <strong>Syft Version:</strong> $server_version<br />

            </div>
            <div class='syft-alert-info syft-space'>
                &#9432;&nbsp;
                This datasite is run by the library PySyft to learn more about how it works visit
                <a href="https://github.com/OpenMined/PySyft">github.com/OpenMined/PySyft</a>.
            </div>
            <h4>Commands to Get Started</h4>
            $command_list
        </div><br />
        """

# json schema primitive types
primitive_mapping = {
    list: "array",
    bool: "boolean",
    int: "integer",
    type(None): "null",
    float: "number",
    dict: "object",
    str: "string",
}


def make_fake_type(_type_str: str) -> dict[str, Any]:
    jsonschema: dict[str, Any] = {}
    jsonschema["title"] = _type_str
    jsonschema["type"] = "object"
    jsonschema["properties"] = {}
    jsonschema["additionalProperties"] = False
    return jsonschema


def get_type_mapping(_type: type) -> str:
    if _type in primitive_mapping:
        return primitive_mapping[_type]
    return _type.__name__


def get_types(cls: type, keys: list[str]) -> dict[str, type] | None:
    types = []
    for key in keys:
        _type = None
        if key in cls.__annotations__:
            _type = cls.__annotations__[key]
        else:
            for parent_cls in cls.mro():
                annotations = getattr(parent_cls, "__annotations__", None)
                if annotations and key in annotations:
                    _type = annotations[key]
        if _type is None:
            # print(f"Failed to find type for key: {key} in {cls}")
            return None
        types.append(_type)
    return dict(zip(keys, types))


def convert_attribute_types(
    cls: type, attribute_list: list[str], attribute_types: list[type]
) -> dict[str, Any]:
    jsonschema: dict[str, Any] = {}
    jsonschema["title"] = cls.__name__
    jsonschema["type"] = "object"
    jsonschema["properties"] = {}
    jsonschema["additionalProperties"] = False
    for attribute, _type in dict(zip(attribute_list, attribute_types)).items():
        attribute_dict = {}
        attribute_dict["type"] = get_type_mapping(_type)
        jsonschema["properties"][attribute] = attribute_dict
    jsonschema["required"] = list(attribute_list)
    return jsonschema


def process_type_bank(type_bank: dict[str, tuple[Any, ...]]) -> dict[str, dict]:
    # first pass gets each type into basic json schema format
    json_mappings = {}
    count = 0
    converted_types: dict[str, int] = defaultdict(int)
    for k in type_bank:
        count += 1
        t = type_bank[k]
        (
            nonrecursive,
            serialize,
            deserialize,
            attribute_list,
            exclude_attrs,
            serde_overrides,
            cls,
            attribute_types,
        ) = t

        lib = cls.__module__.split(".")[0]

        # process types with an attribute list and attribute types first
        if attribute_list and attribute_types:
            if ".uid." not in str(cls):
                print(f"Skipping {k}")
                continue
            try:
                print(f"Processing {k}")
                schema = convert_attribute_types(cls, attribute_list, attribute_types)
                json_mappings[cls.__name__] = schema
                converted_types[lib] += 1
            except Exception:  # nosec
                print(f"Failed to process. {k}")

        # TODO: process other types of serializable objects

    converted = sum(converted_types.values())
    print(f"Finished converting {converted} out of {count}")
    return json_mappings


def resolve_references(json_mappings: dict[str, dict]) -> dict[str, dict]:
    # track second pass generated types
    new_types = {}
    for json_schema in json_mappings.values():
        replace_types = {}
        for attribute, config in json_schema["properties"].items():
            if "type" in config:
                _type_str = config["type"]
                if _type_str in primitive_mapping.values():
                    # no issue with primitive types
                    continue
                else:
                    # if we don't have a type yet its because its not supported
                    # lets create an empty type to satisfy the generation process
                    if _type_str not in json_mappings.keys():
                        reference = make_fake_type(_type_str)
                        new_types[_type_str] = reference

                # if the type is a reference we need to replace its type entry after
                replace_types[attribute] = {
                    "$ref": f"./{SCHEMA_FOLDER}/{_type_str}.json"
                }
        # replace any referenced types
        for k, v in replace_types.items():
            json_schema["properties"][k] = v

    # insert any new types created above into the main dict
    for k, v in new_types.items():
        json_mappings[k] = v

    return json_mappings


@deprecated(
    reason="generate_json_schemas is outdated, #1603 for more info",
)
def generate_json_schemas(output_path: str | None = None) -> None:
    # TODO: should we also replace this with the SyftObjectRegistry?
    json_mappings = process_type_bank(sy.serde.recursive.TYPE_BANK)
    json_mappings = resolve_references(json_mappings)
    if not output_path:
        output_path = os.path.dirname(__file__) + RELATIVE_PATH_TO_FRONTEND
        output_path = os.path.abspath(output_path)
    path = Path(output_path) / SCHEMA_FOLDER
    os.makedirs(path, exist_ok=True)
    for k, v in json_mappings.items():
        with open(path / f"{k}.json", "w") as f:
            f.write(json.dumps(v))
