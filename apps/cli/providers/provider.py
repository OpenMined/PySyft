import json

import terrascript
import terrascript.data  # aws_ami, google_compute_image, ...
import terrascript.provider  # aws, google, ...
import terrascript.resource  # aws_instance, google_compute_instance, ...


class Provider:
    def __init__(self, config):
        self._config = config
        self.tfscript = terrascript.Terrascript()

    def update_script(self):
        with open("main.tf.json", "w") as tfjson:
            json.dump(
                self.tfscript, tfjson, indent=2, sort_keys=False,
            )

    def deploy(self):
        if self.config.app.name == "node":
            self.deploy_node()
        elif self.config.app.name == "network":
            self.deploy_network()

    @property
    def config(self):
        return self._config
