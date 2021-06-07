# stdlib
import json
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

# third party
import terrascript
from terrascript import Module
import terrascript.data as data
import terrascript.provider as provider
import terrascript.resource as resource

# grid relative
from ..tf import ROOT_DIR
from ..tf import Terraform
from ..utils import Config


class Provider:
    def __init__(self, config):
        folder_name = f"{config.provider}-{config.app.name}-{config.app.id}"
        _dir = os.path.join(ROOT_DIR, folder_name)
        os.makedirs(_dir, exist_ok=True)

        self.TF = Terraform(dir=_dir)
        self.tfscript = terrascript.Terrascript()
        self.validated = False

    def validate(self):
        self.TF.write(self.tfscript)
        try:
            self.TF.init()
            self.TF.validate()
            self.validated = True
            return True
        except subprocess.CalledProcessError as err:
            return False

    def deploy(self):
        if not self.validated:
            return (False, {})

        try:
            self.TF.apply()
            output = self.TF.output()
            return (True, output)
        except subprocess.CalledProcessError as err:
            return (False, {"ERROR": err})

    def destroy(self):
        try:
            self.TF.destroy()
            return True
        except subprocess.CalledProcessError as err:
            return False
