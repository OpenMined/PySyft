#!/usr/bin/python3

# stdlib
import glob
import os
import re
import sys


def replace_empty_export(content: str) -> str:
    pattern = r"export interface (\w+) {}"

    def replace_function(match: re.Match) -> str:
        thing = match.group(1)
        return f"import type {{ {thing} }} from './{thing}';"

    return re.sub(pattern, replace_function, content)


def process_file(file_path: str) -> None:
    with open(file_path) as file:
        content = file.read()

    new_content = replace_empty_export(content)

    with open(file_path, "w") as file:
        file.write(new_content)


def scan_and_process_ts_files(folder_path: str) -> None:
    ts_files = glob.glob(os.path.join(folder_path, "*.ts"))
    for file_path in ts_files:
        process_file(file_path)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script_name.py /path/to/your/folder")
        sys.exit(1)

    folder_path = sys.argv[1]
    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a valid directory.")
    else:
        scan_and_process_ts_files(folder_path)
