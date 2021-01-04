# stdlib
import glob
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any
from typing import Pattern

# third party
import jsonlines
import torch

torch_version = torch.__version__
# --------------------------------------
# add exception and it's handler
# --------------------------------------
exception_pattern_1 = re.compile("no attribute|no NVIDIA driver")


def fix_exception_pattern_1(not_available: list, **kwargs: Any) -> None:
    ele = {"lte_version": torch_version, "gte_version": torch_version}
    if ele not in not_available:
        not_available.append(ele)


exception_pattern_2 = re.compile(
    "convert"
    + "|implement"
    + "|support"
    + "|must be Tensor"
    + "|gradients"
    + "|'CPU' backend"
    + "|xpected"
    + "|can't be cast to the desired output type"
    + "|Can only calculate the mean of floating types"
)


def fix_exception_pattern_2(
    not_available: list, tensor_type: str, **kwargs: None
) -> None:
    def get_ele_index() -> int:
        keys = set({"data_types", "lte_version", "gte_version"})
        i = -1
        for i, ele in enumerate(not_available):
            if (
                set(ele) == keys
                and ele["lte_version"] == torch_version
                and ele["gte_version"] == torch_version
            ):
                return i
        not_available.append(
            {
                "data_types": [],
                "lte_version": torch_version,
                "gte_version": torch_version,
            }
        )
        return i + 1

    i = get_ele_index()
    if tensor_type not in not_available[i]["data_types"]:
        not_available[i]["data_types"].append(tensor_type)


exception_fix = []
exception_fix.append((exception_pattern_1, fix_exception_pattern_1))
exception_fix.append((exception_pattern_2, fix_exception_pattern_2))  # type: ignore


# ------------------------------
# some helper function
def match_pattern(exception: str, pattern: Pattern) -> int:
    return len(pattern.findall(exception)) > 0


def is_failed_op(line: str, failed_ops: set) -> bool:
    for op in failed_ops:
        if op in line:
            return True
    return False


# --------------------------------
# backup
p = Path("./src/syft/lib/torch/")
shutil.copyfile(p / "allowlist.py", p / "allowlist.py.bak")
shutil.copyfile(
    "./tests/syft/lib/allowlist_test.json", "./tests/syft/lib/allowlist_test.json.bak"
)


# -------------------------------
# loop:
# [run slow test] -> [comment out allowlist.py] -> [fix allowlist_test.json] -> [run slow test] -> ...
pre_failed_ops = None
same_fail_count = 0
loop_cnt = 0
continue_loop = True
while continue_loop:
    print()
    print("-" * 20 + f"loop {loop_cnt}" + "-" * 20)
    loop_cnt = loop_cnt + 1

    # run slow test
    print("Running slow test ...This may take a while.")
    os.system("pytest -m slow -n auto -p no:benchmark --no-cov > /dev/null")
    print("Slow test done.")
    print()

    # Is errors.jsonl file there?
    err_jsonl = glob.glob("./allowlist_test_errors_*.jsonl")
    # if no, all tests pass, stop loop
    if len(err_jsonl) == 0:
        print()
        print("-" * 20)
        print("All tests passed. Auto fix done.")
        print("Don't forget to:")
        print(
            "{:<10}{:<}".format(
                "", f"- recover the original content of {p/'allowlist.py'}"
            )
        )
        print(
            "{:<10}{:<}".format("", f"- remove the backup file {p/'allowlist.py.bak'}")
        )
        print(
            "{:<10}{:<}".format(
                "", "- remove the backup file tests/syft/lib/allowlist_test.json.bak"
            )
        )
        break

    # get errors.jsonl file name
    err_jsonl = err_jsonl[0]  # type: ignore

    # get all the operators that failed
    with open(err_jsonl, "r+", encoding="utf8") as f:  # type: ignore
        failed_ops = set([item["input"]["op_name"] for item in jsonlines.Reader(f)])

    print(f"{len(failed_ops)} operators failed:")
    for cnt, op in enumerate(failed_ops):
        print(f"{cnt:>10}:{op:>20}")

    # in case some operators always fail, this loop may go forever
    # we check if it's always the same group of operators that fails for many loops
    if failed_ops == pre_failed_ops:
        same_fail_count = same_fail_count + 1
    else:
        same_fail_count = 0
    if same_fail_count > 5:
        print(f"The same group of operators fail for the last 5 loops. {failed_ops}")
        continue_loop = False
    pre_failed_ops = failed_ops

    # read original content of allowlist.py
    with open(p / "allowlist.py.bak", "r") as f:
        lines = f.readlines()

    # comment all lines but these contain err_ops
    adding_hash = False
    for i, line in enumerate(lines):
        if line.startswith('allowlist["torch'):
            if not is_failed_op(line, failed_ops):
                adding_hash = True
            else:
                adding_hash = False
        if adding_hash:
            lines[i] = "#" + line

    # overwrite content of allowlist.py
    with open(p / "allowlist.py", "w") as f:
        f.writelines(lines)

    # read allowlist_test.json
    with open("./tests/syft/lib/allowlist_test.json", "r") as f:
        allowlist_test = json.load(f)

    # read error.jsonl
    with open(err_jsonl, "r+", encoding="utf8") as f:  # type: ignore
        # process each error line
        for j, err in enumerate(jsonlines.Reader(f)):
            # get error informations
            op_name = err["input"]["op_name"]
            tensor_type = err["input"]["tensor_type"]
            exception = err["exception"]

            # get the not_available list for this operator
            not_available = []
            if "not_available" in allowlist_test["tests"]["torch.Tensor"][op_name]:
                not_available = allowlist_test["tests"]["torch.Tensor"][op_name][
                    "not_available"
                ]

            # fix this error
            matched = False
            for exception_pattern, fix_method in exception_fix:
                if match_pattern(exception, exception_pattern):
                    fix_method(not_available=not_available, tensor_type=tensor_type)
                    matched = True
            # we suppose to handle all kinds of error, if there is an error we don't handle, raise an exception
            if not matched:
                continue_loop = False
                print(
                    f"A failure can't be handled. Line {j+1} in {err_jsonl}: {exception}"
                )

            # update the not_available
            allowlist_test["tests"]["torch.Tensor"][op_name][
                "not_available"
            ] = not_available

    # update allowlist_test.json
    with open("./tests/syft/lib/allowlist_test.json", "w") as f:
        json.dump(allowlist_test, f, indent=2)
