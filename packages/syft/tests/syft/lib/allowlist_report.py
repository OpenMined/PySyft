# stdlib
import glob
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any
from typing import Dict
from typing import List
from typing import Union

# third party
from jinja2 import Template
from packaging import version

# this forces the import priority to use site-packages first and current dir last
# this allows us to import torch when calling this file directly since there is a
# subdir here also called torch
del sys.path[0]
sys.path.append("")
# third party
import torch as th  # noqa: E402

# syft absolute
from syft.lib.torch import allowlist  # noqa: E402

TORCH_VERSION = version.parse(th.__version__.split("+")[0])
py_ver = sys.version_info
PYTHON_VERSION = version.parse(f"{py_ver.major}.{py_ver.minor}")
OS_NAME = platform.system().lower()

# we need a file to keep all the errors in that makes it easy to debug failures
TARGET_PLATFORM = f"{PYTHON_VERSION}_{OS_NAME}"
REPORT_FILE_PATH = os.path.abspath(
    Path(__file__) / "../../../.." / f"allowlist_report_{TARGET_PLATFORM}.html"
)

report_path = os.path.abspath((Path(__file__) / "../../../.."))
support_files = glob.glob(os.path.join(report_path, "allowlist_test_support_*.jsonl"))

if len(support_files) < 1:
    print("Generate allowlist_test_support files first.")
    sys.exit(1)

# complex have been removed for now as they are rare and have some known bugs
# qints have been disabled for now and are added as a separate ticket
dtypes = [
    "bool",
    "uint8",
    "int8",
    "int16",
    "int32",
    "int64",
    "float16",
    "bfloat16",
    "float32",
    "float64",
    # "complex32",
    # "complex64",
    # "complex128",
    # "qint8",
    # "quint8",
    # "qint32",
]

# 1.4.0 has been temporarily disabled and will be re-investigated
torch_versions = ["1.6.0", "1.7.1", "1.8.0"]


# this handles instances where the allow list provides more meta information
def get_return_type(support_dict: Union[str, Dict[str, str]]) -> str:
    if isinstance(support_dict, str):
        return support_dict
    else:
        return support_dict["return_type"]


BASIC_OPS = list()
BASIC_OPS_RETURN_TYPE = {}
# here we are loading up the true allowlist which means that we are only testing what
# can be used by the end user
for method, return_type_name_or_dict in allowlist.items():
    if method.startswith("torch.Tensor."):
        return_type = get_return_type(support_dict=return_type_name_or_dict)
        method_name = method.split(".")[-1]
        BASIC_OPS.append(method_name)
        BASIC_OPS_RETURN_TYPE[method_name] = return_type

ops: Dict[str, Any] = {}

# these are all the expected ops
for op in BASIC_OPS:
    ops[op] = {"dtypes": {}}
    for torch_version in torch_versions:
        ops[op]["dtypes"][torch_version] = {}
        for dtype in dtypes:
            ops[op]["dtypes"][torch_version][dtype] = {
                "status": "untested",
                "num_pass": 0,
                "num_fail": 0,
                "num_skip": 0,
                "num_not_available": 0,
            }


def parse_filename_versions(file_path: str) -> List[str]:
    filename = os.path.basename(file_path)
    return filename.replace("allowlist_test_support_", "").split("_")


# parse the data by reading every line in and building a dict
for support_file in support_files:
    with open(support_file, "r") as f:
        versions = parse_filename_versions(support_file)
        python_version = versions[0]
        torch_version = versions[1]
        os_name = versions[2]
        for line in f.readlines():
            test_run = json.loads(line)
            op_name = test_run["op_name"]
            dtype = test_run["tensor_type"]
            status = test_run["status"]
            if op_name not in ops:
                print(f"op {op_name} not found in main ops list")
                continue

            if dtype not in ops[op_name]["dtypes"][torch_version]:
                print(f"dtype {dtype} not found in {torch_version} main ops list")
            else:
                # here we have many repeat tests for the same op and dtype, for now lets
                # mark them as either skip, pass, or fail where failure takes highest
                # priority, then pass, then skip, meaning a single failure will mark
                # the whole cell, with no failure a single pass wil mark as pass
                # and if nothing but skip, it will be marked skip

                # lets count them for later
                if status in ["pass", "fail", "skip", "not_available"]:
                    key = f"num_{status}"
                    ops[op_name]["dtypes"][torch_version][dtype][key] += 1
                    # recalculate a rough ratio
                    num_pass = ops[op_name]["dtypes"][torch_version][dtype]["num_pass"]
                    num_fail = ops[op_name]["dtypes"][torch_version][dtype]["num_fail"]
                    ratio = num_pass + 1 / (num_pass + num_fail + 1)
                    ops[op_name]["dtypes"][torch_version][dtype]["majority"] = (
                        ratio >= 0.5 and num_pass > 0
                    )

                current_status = ops[op_name]["dtypes"][torch_version][dtype]["status"]

                if status == "fail":
                    # set fail
                    ops[op_name]["dtypes"][torch_version][dtype]["status"] = "fail"
                elif status == "pass" and current_status != "fail":
                    # set pass
                    ops[op_name]["dtypes"][torch_version][dtype]["status"] = "pass"
                elif status == "not_available" and current_status == "untested":
                    # set not_available
                    ops[op_name]["dtypes"][torch_version][dtype][
                        "status"
                    ] = "not_available"
                elif status == "skip" and current_status == "not_available":
                    # set skip
                    ops[op_name]["dtypes"][torch_version][dtype]["status"] = "skip"

with open(__file__.replace(".py", ".j2"), "r") as f:
    tm = Template(f.read())
    report_html = tm.render(dtypes=dtypes, torch_versions=torch_versions, ops=ops)
    with open(REPORT_FILE_PATH, "w+") as f:
        f.write(report_html)

print("\nPySyft Torch Compatibility Report Created:")
print(REPORT_FILE_PATH)
