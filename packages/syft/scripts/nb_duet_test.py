#!/usr/bin/env python3


# For creating Duet tests from notebooks, adapt your notebooks to the following rules:
# 1. The DO/DS notebooks have a predefined format:
#  <test_name>_Data_Owner.ipynb and <test_name>_Data_Scientist.ipynb.
#
# 2. Each notebook must have special markdown cells called Checkpoints, with the
#    following properties:
#     - The cell begins with the text "Checkpoint <int>: "
#       used by the script for extracting the step number.
#     - Each Checkpoint cell acts as a barrier in the generated scripts.
#     - In the DO tests, getting to Checkpoint N means creating a file to mark the
#       checkpoint and waiting for the DS instance to get the Checkpoint N as well.
#     - In the DS tests, getting to Checkpoint N means creating a file to mark the
#       checkpoint and waiting for the DO instance to get to Checkpoint N + 1.
#     - The DS notebook instance should always end with a checkpoint cell, to wait for
#       the DO instance to assert the results.

# stdlib
from collections import defaultdict
import json
import os
from pathlib import Path
import re
import shutil
from typing import List

# third party
from nbconvert import PythonExporter
from nbconvert.writers import FilesWriter
import nbformat


def generate_targets() -> List[Path]:
    result = []

    for path in PATHS:
        opt_notebooks = list(Path(path).rglob("*.ipynb"))
        for _path in opt_notebooks:
            if ".ipynb_checkpoints" in str(_path):
                continue
            result.append(_path)

    result.sort()
    return result


asyncio_event_loop = """
import os
import time
import asyncio
from pathlib import Path
from . import get_global_var
loop = asyncio.get_event_loop()
"""


checkpoint_template = """
Path(\"{}\").touch()
for retry in range({}):
    if Path(\"{}\").exists():
        break
    task = loop.create_task(asyncio.sleep(1))
    loop.run_until_complete(task)
"""


checkpoint_ack_template = """
Path(\"{}\").touch()
for retry in range({}):
    if Path(\"{}\").exists():
        break
    task = loop.create_task(asyncio.sleep(1))
    loop.run_until_complete(task)
assert Path(\"{}\").exists()
"""

PATHS = [
    "examples/data-science/pandas",
    "examples/differential-privacy",
    "examples/duet/dcgan",
    "examples/duet/mnist_lightning",
    "examples/duet/mnist",
    "examples/duet/reinforcement_learning",
    "examples/duet/super_resolution",
    "examples/duet/time_sequence_prediction",
    "examples/homomorphic-encryption",
    "examples/private-ai-series/duet_basics",
    "examples/private-ai-series/duet_iris_classifier",
]

NOTEBOOK_TESTS_PATH = Path("tests/syft/notebooks")
CHECKPOINT_PATH = Path("tests/syft/notebooks/checkpoints")
TARGETS = generate_targets()


def generate_duet_notebooks() -> None:
    tests = defaultdict(list)

    SLEEP_TIME = 500

    try:
        os.makedirs(NOTEBOOK_TESTS_PATH, exist_ok=True)
    except BaseException as e:
        print("os.makedirs failed ", e)

    try:
        shutil.rmtree(CHECKPOINT_PATH)
    except BaseException as e:
        print("rmtree failed ", e)

    try:
        os.makedirs(CHECKPOINT_PATH, exist_ok=True)
    except BaseException as e:
        print("os.makedirs failed ", e)

    testcase_lib = {}

    for path in TARGETS:
        testname = re.sub("[^0-9a-zA-Z]+", "_", str(path))
        output = NOTEBOOK_TESTS_PATH / testname

        file_name = str(path.stem)
        is_do = False
        is_ds = False

        if file_name.endswith("_Data_Scientist"):
            testcase = file_name.replace("_Data_Scientist", "")
            tests[testcase].append(testname)
            is_ds = True
        elif file_name.endswith("_Data_Owner"):
            testcase = file_name.replace("_Data_Owner", "")
            tests[testcase].append(testname)
            is_do = True
        else:
            continue

        load_lib_search = r"load_lib\(\W+([a-z_-]+)\W+\)"

        with open(path, "r") as f:
            load_lib_results = re.search(load_lib_search, str(f.read()), re.IGNORECASE)
            if load_lib_results:
                lib_name = load_lib_results.group(1)
                testcase_lib[testcase] = lib_name

        notebook_nodes = nbformat.read(path, as_version=4)

        custom_cell = nbformat.v4.new_code_cell(source=asyncio_event_loop)
        notebook_nodes["cells"].insert(0, custom_cell)

        for idx, cell in enumerate(notebook_nodes["cells"]):
            if cell["cell_type"] == "code" and "loopback=True" in cell["source"]:
                network_url = "f'http://0.0.0.0:{get_global_var()}'"
                notebook_nodes["cells"][idx]["source"] = cell["source"].replace(
                    "loopback=True", f"loopback=True, network_url={network_url}"
                )
            if cell["cell_type"] == "markdown" and "Checkpoint" in cell["source"]:
                checkpoint = (
                    cell["source"]
                    .lower()
                    .split("checkpoint")[1]
                    .strip()
                    .split(":")[0]
                    .strip()
                )

                testcase_checkpoint_dir = f"checkpoints/{testcase}/"

                # For DO, we wait until DS gets to the same checkpoint
                if is_do:
                    ck_file = testcase_checkpoint_dir + (
                        testcase + "_DO_checkpoint_" + str(checkpoint)
                    )
                    wait_file = testcase_checkpoint_dir + (
                        testcase + "_DS_checkpoint_" + str(checkpoint)
                    )
                    checkpoint_cell = nbformat.v4.new_code_cell(
                        source=checkpoint_template.format(
                            ck_file, SLEEP_TIME, wait_file
                        )
                    )

                # For DS, we wait until DO gets to the next checkpoint
                elif is_ds:
                    ck_file = testcase_checkpoint_dir + (
                        testcase + "_DS_checkpoint_" + str(checkpoint)
                    )
                    wait_file = testcase_checkpoint_dir + (
                        testcase + "_DO_checkpoint_" + str(int(checkpoint) + 1)
                    )
                    checkpoint_cell = nbformat.v4.new_code_cell(
                        source=checkpoint_ack_template.format(
                            ck_file, SLEEP_TIME, wait_file, wait_file
                        )
                    )
                notebook_nodes["cells"][idx] = checkpoint_cell

        try:
            exporter = PythonExporter()

            (body, resources) = exporter.from_notebook_node(notebook_nodes)
            write_file = FilesWriter()

            # replace empty cells with print statements for easy debugging
            empty_cell = "# In[ ]:"
            counter = 1
            cell_type = "DO" if is_do else "DS"
            while empty_cell in body:
                body = body.replace(
                    empty_cell, f"print('{cell_type} Cell: {counter}')", 1
                )
                counter += 1

            # replace any test variables / lines to make things faster in test mode
            json_file = f"{path}.json"
            if os.path.exists(json_file):
                with open(json_file, "r") as f:
                    json_rules = json.loads(f.read())

                    for rules in json_rules["replace_lines"]:
                        try:
                            body = re.sub(
                                rules["match"],
                                rules["replace"],
                                body,
                                flags=re.MULTILINE,
                            )
                        except Exception as e:
                            print(
                                f"Failed to replace rule {rules} for test: {path}. {e}"
                            )

            write_file.write(
                output=body, resources=resources, notebook_name=str(output)
            )
        except Exception as e:
            print(f"There was a problem exporting the file(s): {e}")

    for case in tests:
        test = tests[case]
        if len(test) != 2:
            print("invalid testcase ", test)

        print(case, test)

        template = open(NOTEBOOK_TESTS_PATH / "duet_test.py.template").read()

        output_py = template.replace("{{TESTCASE}}", str(case))
        output_py = output_py.replace("checkpoints", f"checkpoints/{case}")

        for script in test:
            if "Data_Owner" in script:
                output_py = output_py.replace("{{DO_SCRIPT}}", script)
            elif "Data_Scientist" in script:
                output_py = output_py.replace("{{DS_SCRIPT}}", script)

        decorator = ""
        if case in testcase_lib:
            lib_name = testcase_lib[case]
            decorator = f"@pytest.mark.vendor(lib='{lib_name}')"
        output_py = output_py.replace(
            "{{LIB_DECORATOR}}", decorator + "\n@pytest.mark.duet"
        )

        with open(NOTEBOOK_TESTS_PATH / f"duet_{case}_test.py", "w") as out_py:
            out_py.write(output_py)


if __name__ == "__main__":
    generate_duet_notebooks()
