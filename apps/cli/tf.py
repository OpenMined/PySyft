import subprocess


class Terraform:
    def __init__(self):
        super().__init__()

    def init(self):
        return subprocess.call("terraform init", shell=True,)

    def validate(self):
        return subprocess.call("terraform validate", shell=True,)

    def plan(self):
        return subprocess.call("terraform plan", shell=True,)

    def apply(self):
        return subprocess.call("terraform apply", shell=True,)

    def destroy(self):
        return subprocess.call("terraform destroy", shell=True,)


TF = Terraform()
