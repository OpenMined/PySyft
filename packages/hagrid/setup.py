# stdlib
import os
import subprocess

# third party
from setuptools import find_packages
from setuptools import setup

hagrid_path = os.path.abspath(os.path.join(os.path.realpath(__file__), "../../hagrid/"))

syft_path = os.path.abspath(os.path.join(os.path.realpath(__file__), "../../syft/"))

grid_path = os.path.abspath(os.path.join(os.path.realpath(__file__), "../../grid/"))


print("\n\n")
print("PREPARING INSTALLATION:")
print("\n")
cmd = "rm -rf build; rm -rf dist; rm -rf grid; rm -rf syft;"
print(cmd)
subprocess.call(cmd, shell=True)
cmd = "cp -r " + syft_path + " " + hagrid_path + "/"
print(cmd)
subprocess.call(cmd, shell=True)

cmd = "cp -r " + grid_path + " " + hagrid_path + "/"
print(cmd)
subprocess.call(cmd, shell=True)
print("\n\n")


def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            path = os.path.join("..", path, filename)
            _path = str(path)
            if ".pyc" not in _path and "build/" not in _path:
                paths.append(path)
    return paths


extra_files = package_files("./syft")
extra_files = package_files("./grid")


setup(
    name="hagrid",
    description="Happy Automation for Grid",
    version="0.1",
    author="Andrew Trask <andrew@openmined.org>",
    packages=find_packages(),
    install_requires=[
        "click",
        "names",
        "requests",
        "setuptools",
        "ansible",
        "ansible-core",
        "paramiko",
        "ascii_magic",
    ],
    package_data={"": extra_files},
    include_package_data=True,
    entry_points={"console_scripts": ["hagrid = hagrid.cli:cli"]},
)
