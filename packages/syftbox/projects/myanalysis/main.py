# /// script
# dependencies = [
#    "opendp==0.11.1",
#    "syftbox==0.1.0",
#    "pandas==2.2.3",
# ]
#
# [tool.uv.sources]
# syftbox = { path = "/Users/madhavajay/dev/syft", editable = true }
# ///

__name__ = "myanalysis"
__author__ = "david.rolle@gmail.com"


def input_reader(private: bool = False):
    import pandas as pd

    from syftbox.lib import sy_path

    inputs = {}
    inputs["trade_data"] = pd.read_csv(sy_path("./inputs/trade_data/trade_mock.csv", resolve_private=private))
    return inputs


def output_writer(result, private: bool = False):
    import json

    output_path = "./output/result/result.json"
    if not private:
        output_path = output_path.replace(".json", ".mock.json")
    with open(output_path, "w") as f:
        f.write(json.dumps(result))


# START YOUR CODE


def myanalysis(trade_data):
    import opendp.prelude as dp

    dp.enable_features("contrib")

    aggregate = 0.0
    base_lap = dp.m.make_laplace(
        dp.atom_domain(T=float),
        dp.absolute_distance(T=float),
        scale=5.0,
    )
    noise = base_lap(aggregate)

    total = trade_data["Trade Value (US$)"].sum()
    result = (float(total / 1_000_000), float(noise))

    print(result)
    if result[0] > 3:
        print("Got mock")
    else:
        print("Got private")
    return result


# END YOUR CODE


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Process some input.")
    parser.add_argument("--private", action="store_true", help="Run in private mode")
    args = parser.parse_args()

    print(f"Running: {__name__} from {__author__}")
    inputs = input_reader(private=args.private)
    print("> Reading Inputs", inputs)

    output = myanalysis(**inputs)

    print("> Writing Outputs", output)
    output_writer(output, private=args.private)
    print(f"> ✅ Running {__name__} Complete!")


main()
