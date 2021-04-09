# %load PySyft/scripts/autofix_allowlist_test.py
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
has_cuda = torch.cuda.is_available()
# --------------------------------------
# add exception and it's handler
# --------------------------------------
exception_pattern_1 = re.compile(
    "no attribute" + "|no NVIDIA driver" + "|Torch not compiled with CUDA enabled"
)


def fix_exception_pattern_1(not_available: list, **kwargs: Any) -> None:
    ele = {
        "lte_version": torch_version,
        "gte_version": torch_version,
        "reason": "no_cpu",
    }
    if ele not in not_available:
        not_available.append(ele)


# inappropriate _arg("inputs" in allowlist_test.json)
exception_pattern_args = re.compile(
    "argument '\w+.*\(position 1\) must be \w+, not \w+.*\[torch\]"
)


def fix_exception_pattern_args(
    not_available: list, inputs: Any, **kwargs: None
) -> None:
    def get_ele_index() -> int:

        keys = {"inputs", "lte_version", "gte_version", "reason"}
        i = -1
        for i, ele in enumerate(not_available):
            if (
                set(ele) == keys
                and ele["lte_version"] == torch_version
                and ele["gte_version"] == torch_version
                and ele["reason"] == "bad_input"
            ):
                return i
        not_available.append(
            {
                "inputs": [],
                "reason": "bad_input",
                "lte_version": torch_version,
                "gte_version": torch_version,
            }
        )
        return i + 1

    def exactly_eq(a: Any, b: Any) -> bool:
        """
        `True==1` and `False==0` will return True;
        But we want them be False, so we also check if they are of the same type.
        """
        return type(a) == type(b) and a == b

    i = get_ele_index()
    already_exists = sum([exactly_eq(inputs, _) for _ in not_available[i]["inputs"]])
    if not already_exists:
        not_available[i]["inputs"].append(inputs)


# exceptions thrown by torch, tagged by allowlist_test.py as "[torch]"
exception_pattern_2 = re.compile("\[torch\]")


def fix_exception_pattern_2(
    not_available: list, tensor_type: str, inputs: Any, cuda: bool, **kwargs: None
) -> None:
    def get_ele_index() -> int:
        keys = {"data_types", "lte_version", "gte_version", "cuda"}
        i = -1
        for i, ele in enumerate(not_available):
            if (
                set(ele) == keys
                and ele["lte_version"] == torch_version
                and ele["gte_version"] == torch_version
                and ele["cuda"] == cuda
            ):
                return i
        not_available.append(
            {
                "data_types": [],
                "cuda": cuda,
                "lte_version": torch_version,
                "gte_version": torch_version,
            }
        )
        return i + 1

    i = get_ele_index()
    if tensor_type not in not_available[i]["data_types"]:
        not_available[i]["data_types"].append(tensor_type)


# exceptions thrown by syft
# if something new that's not mentioned here happens, "A failure can't be handled..." appears
exception_pattern_syft = re.compile(
    "reshape is not implemented for sparse tensors"
    + "|aten::empty_strided"
    + "|If you are using DistributedDataParallel \(DDP\) for training"
    + "|not present in the AST"
    + "|Can't detach views in-place\. Use detach\(\) instead"
    + "|object has no attribute '__module__'"
)


def fix_exception_pattern_syft(
    not_available: list, inputs: Any, cuda: bool, **kwargs: None
) -> None:
    def get_ele_index() -> int:
        keys = {"lte_version", "gte_version", "cuda", "reason"}
        i = -1
        for i, ele in enumerate(not_available):
            if (
                set(ele) == keys
                and ele["lte_version"] == torch_version
                and ele["gte_version"] == torch_version
                and ele["cuda"] == cuda
                and ele["reason"] == "not_supported_syft"
            ):
                return i
        not_available.append(
            {
                "cuda": cuda,
                "reason": "not_supported_syft",
                "lte_version": torch_version,
                "gte_version": torch_version,
            }
        )
        return i + 1

    i = get_ele_index()


exception_fix = []
exception_fix.append((exception_pattern_1, fix_exception_pattern_1))
exception_fix.append((exception_pattern_args, fix_exception_pattern_args))  # type: ignore
exception_fix.append((exception_pattern_2, fix_exception_pattern_2))  # type: ignore
exception_fix.append((exception_pattern_syft, fix_exception_pattern_syft))  # type: ignore

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
root_dir = os.path.abspath(Path(os.path.dirname(__file__)) / "..")
p = Path(f"{root_dir}/src/syft/lib/torch/")
shutil.copyfile(p / "allowlist.py", p / "allowlist.py.bak")
shutil.copyfile(
    f"{root_dir}/tests/syft/lib/allowlist_test.json",
    f"{root_dir}/tests/syft/lib/allowlist_test.json.bak",
)

# temporarily clean up the "skip" and "not_available" rule,
with open(f"{root_dir}/tests/syft/lib/allowlist_test.json", "r") as f:
    allowlist_test = json.load(f)
skip_no_cuda_exists = False
for op in allowlist_test["tests"]["torch.Tensor"].keys():
    if "skip" in allowlist_test["tests"]["torch.Tensor"][op]:
        for s in allowlist_test["tests"]["torch.Tensor"][op]["skip"]:
            # just reminds users that it has not been tested under CUDA
            if s["reason"] == "skip_no_cuda":
                skip_no_cuda_exists = True
                if has_cuda:
                    allowlist_test["tests"]["torch.Tensor"][op]["skip"].remove(s)

            elif s["reason"] != "untested":
                # "input_quant" etc.
                allowlist_test["tests"]["torch.Tensor"][op]["skip"].remove(s)

            else:
                pass

    else:
        allowlist_test["tests"]["torch.Tensor"][op]["skip"] = []

    if not skip_no_cuda_exists and not has_cuda:
        allowlist_test["tests"]["torch.Tensor"][op]["skip"] += [
            {
                "lte_version": torch_version,
                "gte_version": torch_version,
                "reason": "skip_no_cuda",
            }
        ]

    if "not_available" in allowlist_test["tests"]["torch.Tensor"][op]:
        for s in allowlist_test["tests"]["torch.Tensor"][op]["not_available"]:
            # keep "added_feature" "deprecated", remove the rest
            if "reason" in s and s["reason"] in ["added_feature", "deprecated"]:
                continue
            # these are test results for non-current torch versions.
            if (
                "lte_version" in s
                and "gte_version" in s
                and s["lte_version"] != torch_version
            ):
                continue
            allowlist_test["tests"]["torch.Tensor"][op]["not_available"].remove(s)

with open(f"{root_dir}/tests/syft/lib/allowlist_test.json", "w") as f:
    json.dump(allowlist_test, f, indent=2)

# -------------------------------
# loop:
# [run slow test] -> [comment out allowlist.py] -> [fix allowlist_test.json] -> [run slow test] -> ...
failed_ops = None
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
    # os.system("pytest -m torch -n auto -p no:benchmark --tb=no")
    os.system("pytest -m torch -p no:benchmark --tb=no")
    print("Slow test done.")
    print()

    # Is errors.jsonl file there?
    err_jsonl = glob.glob(f"{root_dir}/allowlist_test_errors_*.jsonl")
    # if no, all tests pass, stop loop
    if len(err_jsonl) == 0:
        print()
        print("-" * 20)
        print("All tests passed. Auto fix done.")
        os.system(f"git co {p/'allowlist.py'}")
        os.system(f"rm {p/'allowlist.py.bak'}")
        os.system("rm tests/syft/lib/allowlist_test.json.bak")
        failed_ops = None
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

    # read allowlist_test.json
    with open(f"{root_dir}/tests/syft/lib/allowlist_test.json", "r") as f:
        allowlist_test = json.load(f)

    # read error.jsonl
    with open(err_jsonl, "r+", encoding="utf8") as f:  # type: ignore
        # process each error line
        for j, err in enumerate(jsonlines.Reader(f)):
            # get error informations
            op_name = err["input"]["op_name"]
            tensor_type = err["input"]["tensor_type"]
            inputs = err["input"]["_args"]
            exception = err["exception"]
            cuda = err["input"]["cuda"]

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
                    fix_method(
                        not_available=not_available,
                        tensor_type=tensor_type,
                        inputs=inputs,
                        cuda=cuda,
                    )
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
    with open(f"{root_dir}/tests/syft/lib/allowlist_test.json", "w") as f:
        json.dump(allowlist_test, f, indent=2)


if failed_ops is not None:
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
            # don't comment out `sum` and `backward`, because we will need them when test "grad"
            if "sum" in line or "backward" in line:
                adding_hash = False
        if adding_hash:
            lines[i] = "#" + line

    # overwrite content of allowlist.py
    with open(p / "allowlist.py", "w") as f:
        f.writelines(lines)


# read allowlist_test.json
with open(f"{root_dir}/tests/syft/lib/allowlist_test.json", "r") as f:
    allowlist_test = json.load(f)

# optimize json file
for op, config in allowlist_test["tests"]["torch.Tensor"].items():
    lte_key = "lte_version"
    gte_key = "gte_version"

    # look at not available rules
    if "not_available" in config:
        na_rules = config["not_available"]
        new_rule = None
        for rule in na_rules:
            if lte_key in rule and gte_key in rule:
                rule_lte_version = rule[lte_key]
                rule_gte_version = rule[gte_key]
                if (
                    rule_lte_version == torch_version
                    and rule_gte_version == torch_version
                ):
                    new_rule = rule

        old_rule = None
        match_found = False
        # found a new rule to optimize
        if new_rule is not None:
            for rule in na_rules:
                if new_rule != rule:
                    old_rule_copy = rule.copy()
                    old_rule_copy.pop(lte_key, None)
                    old_rule_copy.pop(gte_key, None)
                    new_rule_copy = new_rule.copy()
                    new_rule_copy.pop(lte_key, None)
                    new_rule_copy.pop(gte_key, None)
                    if old_rule_copy.keys() == new_rule_copy.keys():
                        for k in old_rule_copy.keys():
                            if isinstance(old_rule_copy[k], list) and isinstance(
                                new_rule_copy[k], list
                            ):
                                old_rule_copy[k] = sorted(
                                    old_rule_copy[k], key=lambda x: str(x)
                                )
                                new_rule_copy[k] = sorted(
                                    new_rule_copy[k], key=lambda x: str(x)
                                )

                        # two rules with no version limits and sorted lists match
                        if old_rule_copy == new_rule_copy:
                            match_found = True
                            old_rule = rule.copy()

        # remove the new rule and update the old rule to use current torch_version
        if match_found:
            new_na_rules = []
            for rule in config["not_available"]:
                if rule == new_rule:
                    continue
                if rule == old_rule:
                    rule[lte_key] = torch_version

                new_na_rules.append(rule)
            config["not_available"] = new_na_rules

# update allowlist_test.json
with open(f"{root_dir}/tests/syft/lib/allowlist_test.json", "w") as f:
    json.dump(allowlist_test, f, indent=2)
