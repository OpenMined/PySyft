"""
Running script
1. jupyter nbconvert --to python --execute <Libname>_missing_return/*.ipynb
2. python update.py -l <Libname>

"""
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument(
    "-l", dest="lib", required=True, help="name of the model to be added to ast"
)
args = parser.parse_args()


def main() -> None:

    package_name = args.lib
    PKG_SUPPORT_NAME = f"{package_name}.pkg_support.json"
    i = f"{package_name}_missing_return"
    print(i)
    # module_xd = __import__(i, fromlist = ['object'])
    exec(f"{i} = __import__(i)")
    package_support = dict()
    with open(PKG_SUPPORT_NAME) as f:
        package_support = json.load(f)

    # print(module_xd.__name__)
    # print(exec(f"{module_xd.__name__}.xgboost_sklearn_XGBRFClassifier.type_xgboost_sklearn_XGBRFClassifier_fit"))
    allowlist = package_support["methods"]

    for key in allowlist.keys():
        if allowlist[key] in ["_syft_missing", "_syft_return_absent"]:
            # rewrite allowlist[key] =

            arr = key.split(".")[:-1]

            class_ = str()

            for a in arr:
                class_ += a + "_"

            key_ = key.replace(".", "_")
            # executing this string should work :)
            try:
                # print(f"allowlist[key] = {i}.{class_[:-1]}.type_{key_}")
                # value = "_syft_missing"
                # print(f"Executing value = {i}.{class_[:-1]}.type_{key_}")
                exec(f"allowlist[key] = {i}.{class_[:-1]}.type_{key_}")
                # print(exec(f'{i}.{class_[:-1]}.type_{key_}'))
                if allowlist[key] not in ["_syft_missing", "_syft_return_absent"]:
                    print(f"Updating {allowlist[key]}")
                    # allowlist[key] = value
            except Exception as e:
                print(f"Some Exception in update.py\n\t{e}")

    package_support["methods"] = allowlist

    with open(PKG_SUPPORT_NAME, "w") as outfile:
        json.dump(package_support, outfile)


if __name__ == "__main__":
    main()
