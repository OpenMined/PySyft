"""To be extended in the near future."""
from collections import OrderedDict
import logging
import subprocess

import tf_encrypted as tfe


logger = logging.getLogger("tf_encrypted")


class TFEWorker:
    # TODO(Morten) this should be turned into a proxy, with existing code
    # extracted into a new component that's launched via a script

    def __init__(self, host=None, auto_managed=True):
        self.host = host
        self._server_process = None
        self._auto_managed = auto_managed

    def start(self, player_name, *workers):
        config_filename = "/tmp/tfe.config"

        config, _ = self.config_from_workers(workers)
        config.save(config_filename)

        if self._auto_managed:
            cmd = "python -m tf_encrypted.player --config {} {}".format(
                config_filename, player_name
            )
            self._server_process = subprocess.Popen(cmd.split(" "))
        else:
            logger.info(
                "If not done already, please launch the following "
                "command in a terminal on host '%s':\n"
                "'python -m tf_encrypted.player --config %s %s'\n"
                "This can be done automatically in a local subprocess by "
                "setting `auto_managed=True` when instantiating a TFEWorker.",
                self.host,
                config_filename,
                player_name,
            )

    def stop(self):
        if self._auto_managed:
            if self._server_process is None:
                return
            self._server_process.kill()
            self._server_process.communicate()
            self._server_process = None
        else:
            logger.info("Please terminate the process on host '%s'.", self.host)

    def connect_to_model(self, input_shape, output_shape, *workers):
        config, _ = self.config_from_workers(workers)
        tfe.set_config(config)

        prot = tfe.protocol.SecureNN(
            config.get_player("server0"), config.get_player("server1"), config.get_player("server2")
        )
        tfe.set_protocol(prot)

        self._tf_client = tfe.serving.QueueClient(
            input_shape=input_shape, output_shape=output_shape
        )

        sess = tfe.Session(config=config)
        self._tf_session = sess

    def query_model(self, data):
        self.query_model_async(data)
        return self.query_model_join()

    def query_model_async(self, data):
        self._tf_client.send_input(self._tf_session, data)

    def query_model_join(self):
        return self._tf_client.receive_output(self._tf_session)

    @classmethod
    def config_from_workers(cls, workers):
        if len(workers) != 3:
            raise ValueError("Expected three workers but {} were given".format(len(workers)))

        player_to_worker_mapping = OrderedDict()
        player_to_worker_mapping["server0"] = workers[0]
        player_to_worker_mapping["server1"] = workers[1]
        player_to_worker_mapping["server2"] = workers[2]

        hostmap = OrderedDict(
            [(player_name, worker.host) for player_name, worker in player_to_worker_mapping.items()]
        )
        config = tfe.RemoteConfig(hostmap)

        return config, player_to_worker_mapping
