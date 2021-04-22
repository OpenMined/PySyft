import argparse
import os
from pathlib import Path
import subprocess
import shutil

my_parser = argparse.ArgumentParser(
    description="Generate protobuf messages from a source folder " "to a target folder."
)
my_parser.add_argument(
    "--source",
    action="store",
    type=str,
    required=True,
    help="path to load the " "protobufs",
)
my_parser.add_argument(
    "--target",
    action="store",
    type=str,
    required=True,
    help="path to the " "compiled " "protobufs",
)
my_parser.add_argument(
    "--language",
    action="store",
    type=str,
    required=True,
    help="target " "language for " "protobuf " "compilation",
)
args = my_parser.parse_args()

def python_compile_protobufs(source: str, target: str):
    try:
        shutil.rmtree(target)
    except:
        pass

    source_path = Path(source)
    target_path = Path(target)

    for root, subfolders, files in os.walk(source_path):
        for file in files:
            in_file = Path(root) / file
            out_file = target_path
            os.makedirs(target_path, exist_ok=True)
            subprocess.Popen(["protoc", f"--python_out={out_file}",
                              "--experimental_allow_proto3_optional", f"{in_file}"])

    for root, subfolders, files in os.walk(target_path):
        for file in files:
            if str(file).endswith("_pb2.py"):


def rust_compile_protobufs(source: str, target: str):
    pass


SUPPORTED_TARGET_LANGUAGES = {
    "python3": python_compile_protobufs,
    "rust": rust_compile_protobufs,
}

if __name__ == "__main__":
    if args.language not in SUPPORTED_TARGET_LANGUAGES.keys():
        raise ValueError(
            "Compilation not supported for the specified target language, "
            f"please provide one of: {list(SUPPORTED_TARGET_LANGUAGES.keys())}"
        )

    SUPPORTED_TARGET_LANGUAGES[args.language](args.source, args.target)
