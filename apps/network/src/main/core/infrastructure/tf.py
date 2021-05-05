# stdlib
import json
import os
from pathlib import Path
import subprocess

var = lambda x: "${" + x + "}"
var_module = lambda x, y: var(f"module.{x._name}.{y}")
generate_cidr_block = lambda base_cidr_block, netnum: var(
    f'cidrsubnet("{base_cidr_block}", 8, {netnum})'
)
ROOT_DIR = os.path.join(str(Path.home()), ".pygrid", "api")
# ROOT_DIR = os.path.join("/home/ubuntu/", ".pygrid", "api")


class Terraform:
    def __init__(self, dir: str) -> None:
        super().__init__()
        self.dir = dir

    def write(self, tfscript):
        # save the terraform configuration files
        with open(f"{self.dir}/main.tf.json", "w") as tfjson:
            json.dump(tfscript, tfjson, indent=2, sort_keys=False)

    def init(self):
        return subprocess.run(
            # f"terraform init -input=false -plugin-dir={ROOT_DIR}",
            f"terraform init",
            shell=True,
            cwd=self.dir,
            check=True,
        )

    def validate(self):
        return subprocess.run(
            "terraform validate", shell=True, cwd=self.dir, check=True
        )

    def plan(self):
        return subprocess.run("terraform plan", shell=True, cwd=self.dir, check=True)

    def apply(self):
        return subprocess.run(
            "terraform apply --auto-approve", shell=True, cwd=self.dir, check=True
        )

    def output(self):
        output = subprocess.run(
            "terraform output -json",
            shell=True,
            cwd=self.dir,
            check=True,
            stdout=subprocess.PIPE,
        )
        return json.loads(output.stdout.decode("utf-8"))

    def destroy(self):
        return subprocess.run(
            "terraform destroy --auto-approve", shell=True, cwd=self.dir, check=True
        )
