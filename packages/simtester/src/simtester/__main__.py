# stdlib
import argparse
import asyncio
import importlib.util
import os

# relative
from .actor import Actor
from .runner import Runner


def load_actor_class(module_path, class_name):
    spec = importlib.util.spec_from_file_location("actor_module", module_path)
    actor_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(actor_module)

    # Find the specific class by name
    obj = getattr(actor_module, class_name, None)
    if obj is None:
        raise ImportError(f"Class `{class_name}` not found in `{module_path}`.")
    if not isinstance(obj, type) or not issubclass(obj, Actor):
        raise TypeError(f"`{class_name}` is not a valid subclass of `Actor`.")
    return obj


def parse_actor_args(args):
    actor_data = []
    for i in range(
        0, len(args), 2
    ):  # Iterate over the pairs of `module_path::class_name` and `count`
        path_class_pair = args[i]
        count = int(args[i + 1])

        if "::" not in path_class_pair:
            raise ValueError(
                f"Invalid format for actor class specification: {path_class_pair}"
            )

        module_path, class_name = path_class_pair.split("::")

        # Resolve the absolute module path
        module_path = os.path.abspath(module_path)

        actor_data.append((module_path, class_name, count))
    return actor_data


def main():
    parser = argparse.ArgumentParser(description="Run the simulation tests")
    parser.add_argument(
        "actor_args",
        nargs="+",
        help="Actor class specifications in the format path/to/file.py::Class count.\n"
        "Example usage: `bigquery/level_0_simtest.py::DataScientist 5 bigquery/level_0_simtest.py::Admin 2`.\n"
        "This will spawn 5 DataScientist actors and 2 Admin actors.",
    )
    args = parser.parse_args()

    actor_specs = parse_actor_args(args.actor_args)

    actor_classes = []
    for module_path, class_name, count in actor_specs:
        try:
            actor_class = load_actor_class(module_path, class_name)
        except (ImportError, TypeError) as e:
            print(e)
            return
        actor_classes.append((actor_class, count))

    # Run the simulation with multiple actor classes
    runner = Runner(actor_classes)
    asyncio.run(runner.start())


if __name__ == "__main__":
    main()
