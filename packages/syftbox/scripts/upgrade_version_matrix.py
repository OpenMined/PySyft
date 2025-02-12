import argparse
import json

from packaging.version import Version

parser = argparse.ArgumentParser("upgrade_version_matrix")
parser.add_argument("upgrade_type", choices=["major", "minor", "patch"])
parser.add_argument("--breaking_changes", action="store_true")

args = parser.parse_args()
print(args.upgrade_type)
print(args.breaking_changes)

with open("../syftbox/server2client_version.json") as json_file:
    version_matrix = json.load(json_file)

versions = list(version_matrix.keys())
versions.sort(key=Version)
last_version = versions[-1]
version_numbers = last_version.split(".")

if args.upgrade_type == "patch":
    if args.breaking_changes:
        raise Exception(
            "Patch upgrades imply no breaking changes. If you have breaking changes please consider a minor version upgrade"
        )
    version_numbers[2] = str(int(version_numbers[2]) + 1)
    new_version = ".".join(version_numbers)
    # new_version = last_version
    version_matrix[new_version] = version_matrix[last_version]
elif args.upgrade_type == "minor":
    version_numbers[1] = str(int(version_numbers[1]) + 1)
    version_numbers[2] = "0"
    new_version = ".".join(version_numbers)
    if args.breaking_changes:
        version_matrix[new_version] = [new_version, ""]
        for version in versions:
            version_range = version_matrix[version]
            if version_range[1] == "":
                version_range[1] = new_version
                version_matrix[version] = version_range
    else:
        version_matrix[new_version] = version_matrix[last_version]

elif args.upgrade_type == "major":
    raise NotImplementedError

with open("../syftbox/server2client_version.json", "w") as json_file:
    # json.dump(version_matrix, json_file, indent=4)
    json_file.write("{\n")
    json_file.write(
        ",\n".join(
            [
                f"""  "{key}": ["{version_range[0]}", "{version_range[1]}"]"""
                for key, version_range in version_matrix.items()
            ]
        )
    )
    json_file.write("\n}\n")
