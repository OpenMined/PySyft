import os
import json
import time
from pathlib import Path

import terrascript
import terrascript.data as data
import terrascript.provider as provider
import terrascript.resource as resource
from terrascript import Module

from ..tf import Terraform
from .utils import *


class Provider:
    def __init__(self):
        self.root_dir = os.path.join(str(Path.home()), ".pygrid", "api")
        os.makedirs(self.root_dir, exist_ok=True)

        self.TF = Terraform()
        self.tfscript = terrascript.Terrascript()

    def deploy(self):
        # save the terraform configuration files
        with open(f"{self.root_dir}/main.tf.json", "w") as tfjson:
            json.dump(self.tfscript, tfjson, indent=2, sort_keys=False)

        self.TF.init(self.root_dir)
        self.TF.apply(self.root_dir)
