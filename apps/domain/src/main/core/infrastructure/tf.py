# stdlib
import json
import os
from pathlib import Path
import platform
import subprocess

var = lambda x: "${" + x + "}"
var_module = lambda x, y: var(f"module.{x._name}.{y}")
generate_cidr_block = lambda base_cidr_block, netnum: var(
    f'cidrsubnet("{base_cidr_block}", 8, {netnum})'
)


class Terraform:
    def __init__(self, dir: str, provider: str) -> None:
        super().__init__()
        self.dir = dir
        self.provider = provider

    def write(self, tfscript):
        # save the terraform configuration files
        with open(f"{self.dir}/main.tf.json", "w") as tfjson:
            json.dump(tfscript, tfjson, indent=2, sort_keys=False)

    def init(self):
        if self.provider == "aws":
            plugin_dir = os.path.dirname(self.dir)
            if self.install_plugins(plugin_dir):
                return subprocess.run(
                    f"terraform init -plugin-dir={plugin_dir}",
                    shell=True,
                    cwd=self.dir,
                    check=True,
                )
            else:
                return False
        else:
            return subprocess.run(
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

    def install_plugins(self, dir):
        _os = str(platform.system()).lower()
        if _os == "":
            raise Exception("Can not determine operating system")
        elif _os == "java":
            raise Exception("Terraform does not support this operating system")

        file_dir = os.path.join(
            dir, f"registry.terraform.io/hashicorp/aws/3.30.0/{_os}_amd64/"
        )
        if not os.path.exists(file_dir):
            return subprocess.run(
                f"""
                echo "Install terraform plugins"
                mkdir -p "registry.terraform.io/hashicorp/aws/3.30.0/{_os}_amd64/"
                wget https://releases.hashicorp.com/terraform-provider-aws/3.30.0/terraform-provider-aws_3.30.0_{_os}_amd64.zip
                unzip terraform-provider-aws_3.30.0_{_os}_amd64.zip -d "registry.terraform.io/hashicorp/aws/3.30.0/{_os}_amd64/"
                """
                if self.provider == "aws"
                else "",
                shell=True,
                cwd=dir,
                check=True,
            )
        else:
            return True
