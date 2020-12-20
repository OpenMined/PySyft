import json
import os
import subprocess
import time
from pathlib import Path

import terrascript
import terrascript.data as data
import terrascript.provider as provider
import terrascript.resource as resource
from terrascript import Module

from ..tf import Terraform
from ..utils import Config


class Provider:
    def __init__(self, app):
        self.root_dir = os.path.join(str(Path.home()), ".pygrid", "api", app)
        os.makedirs(self.root_dir, exist_ok=True)

        self.TF = Terraform()
        self.tfscript = terrascript.Terrascript()

    def deploy(self):
        # save the terraform configuration files
        with open(f"{self.root_dir}/main.tf.json", "w") as tfjson:
            json.dump(self.tfscript, tfjson, indent=2, sort_keys=False)

        try:
            self.TF.init(self.root_dir)
            self.TF.validate(self.root_dir)
            self.TF.apply(self.root_dir)
            output = self.TF.output(self.root_dir)
            return (True, output)
        except subprocess.CalledProcessError as err:
            output = {"ERROR": err}
            return (False, output)
