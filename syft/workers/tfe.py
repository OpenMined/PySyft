"""To be extended in the near future."""
from collections import OrderedDict
import subprocess

import tf_encrypted as tfe


class TFEWorker:
    # TODO(Morten) this should be turned into a proxy, with existing code
    # extracted into a new component that's launched via a script

    def __init__(self, host=None):
        self.host = host
        self._server_process = None

    def start(self, player_name, *workers):

        config_filename = "/tmp/tfe.config"

        config = self._config_from_workers(workers)
        config.save(config_filename)
        
        cmd = "python -m tf_encrypted.player --config {} {}".format(config_filename, player_name)
        self._server_process = subprocess.Popen(cmd.split(' '))

    def stop(self):
        if self._server_process is None:
            return

        self._server_process.kill()
        self._server_process.communicate()
        self._server_process = None

    def connect_to_model(self, input_shape, output_shape, *workers):
        config = self._config_from_workers(workers)
        tfe.set_config(config)

        prot = tfe.protocol.SecureNN(
            config.get_player("server0"),
            config.get_player("server1"),
            config.get_player("server2"),
        )
        tfe.set_protocol(prot)

        self._tf_client = tfe.serving.QueueClient(
            input_shape=input_shape,
            output_shape=output_shape)

        sess = tfe.Session(config=config)
        self._tf_session = sess

    def query_model(self, data):
        return self._tf_client.run(self._tf_session, data)

    def _config_from_workers(self, workers):
        if len(workers) != 3:
            raise ValueError("Expected three workers but {} were given".format(len(workers)))

        player_to_worker_mapping = OrderedDict()
        player_to_worker_mapping['server0'] = workers[0]
        player_to_worker_mapping['server1'] = workers[1]
        player_to_worker_mapping['server2'] = workers[2]

        hostmap = OrderedDict([
            (player_name, worker.host)
            for player_name, worker in player_to_worker_mapping.items()
        ])
        config = tfe.RemoteConfig(hostmap)

        return config
