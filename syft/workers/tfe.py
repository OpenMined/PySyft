"""To be extended in the near future."""
from collections import OrderedDict
import logging
import os
import subprocess
import tempfile

import tf_encrypted as tfe


logger = logging.getLogger("tf_encrypted")
_TMP_DIR = tempfile.gettempdir()


class TFEWorker:
    # TODO(Morten) this should be turned into a proxy, with existing code
    # extracted into a new component that's launched via a script

    def __init__(self, host=None, auto_managed=True):
        self.host = host
        self._server_process = None
        self._auto_managed = auto_managed

    def start(self, player_name, cluster):
        """
        Start the worker as a player in the given cluster.
        Depending on whether the worker was constructed with a host or not
        this may launch a subprocess running a TensorFlow server.
        """

        if self.host is None:
            # we're running using a tfe.LocalConfig which doesn't require us to do anything
            return

        config_filename = os.path.join(_TMP_DIR, "tfe.config")
        config = cluster.tfe_config
        config.save(config_filename)

        launch_cmd = "python -m tf_encrypted.player --config {} {}".format(
            config_filename, player_name
        )
        if self._auto_managed:
            self._server_process = subprocess.Popen(launch_cmd.split(" "))
        else:
            logger.info(
                "If not done already, please launch the following "
                "command in a terminal on host %s: '%s'\n"
                "This can be done automatically in a local subprocess by "
                "setting `auto_managed=True` when instantiating a TFEWorker.\n",
                self.host,
                launch_cmd,
            )

    def stop(self):
        """
        Stop the worker. This will shutdown any TensorFlow server launched
        in `start()`.
        """

        if self.host is None:
            # we're running using a tfe.LocalConfig which doesn't require us to do anything
            return

        if self._auto_managed:
            if self._server_process is None:
                return
            self._server_process.kill()
            self._server_process.communicate()
            self._server_process = None
        else:
            logger.info("Please terminate the process on host '%s'.", self.host)

    def connect_to_model(self, input_shape, output_shape, cluster, sess=None):
        """
        Connect to a TF Encrypted model being served by the given cluster.

        This must be done before querying the model.
        """

        config = cluster.tfe_config
        tfe.set_config(config)

        prot = tfe.protocol.SecureNN(
            config.get_player("server0"), config.get_player("server1"), config.get_player("server2")
        )
        tfe.set_protocol(prot)

        self._tf_client = tfe.serving.QueueClient(
            input_shape=input_shape, output_shape=output_shape
        )

        if sess is None:
            sess = tfe.Session(config=config)
        self._tf_session = sess

    def query_model(self, data):
        """
        Encrypt data and sent it as input to the model being served.

        This will block until a result is ready, and requires that
        a connection to the model has already been established via
        `connect_to_model()`.
        """
        self.query_model_async(data)
        return self.query_model_join()

    def query_model_async(self, data):
        """
        Asynchronous version of `query_model` that will not block until a
        result is ready. Call `query_model_join` to retrive result.

        This requires that a connection to the model has already been
        established via `connect_to_model()`.
        """
        self._tf_client.send_input(self._tf_session, data)

    def query_model_join(self):
        """
        Retrives the result from calling `query_model_async`, blocking until
        ready.
        """
        return self._tf_client.receive_output(self._tf_session)


class TFECluster:
    """
    A TFECluster represents a group of TFEWorkers that are aware about each
    other and collectively perform an encrypted computation.
    """

    def __init__(self, *workers):
        tfe_config, player_to_worker_mapping = self._build_cluster(workers)
        self.tfe_config = tfe_config
        self.player_to_worker_mapping = player_to_worker_mapping

    @property
    def workers(self):
        return list(self.player_to_worker_mapping.values())

    def start(self):
        """
        Start all workers in the cluster.
        """
        # Tell the TFE workers to launch TF servers
        for player_name, worker in self.player_to_worker_mapping.items():
            worker.start(player_name, self)

    def stop(self):
        """
        Stop all workers in the cluster.
        """
        for worker in self.workers:
            worker.stop()

    def _build_cluster(self, workers):
        if len(workers) != 3:
            raise ValueError(f"Expected three workers but {len(workers)} were given")

        player_to_worker_mapping = OrderedDict()
        player_to_worker_mapping["server0"] = workers[0]
        player_to_worker_mapping["server1"] = workers[1]
        player_to_worker_mapping["server2"] = workers[2]

        use_local_config = all(worker.host is None for worker in workers)
        if use_local_config:
            config = tfe.LocalConfig(
                player_names=player_to_worker_mapping.keys(), auto_add_unknown_players=False
            )
            return config, player_to_worker_mapping

        # use tfe.RemoteConfig
        hostmap = OrderedDict(
            [(player_name, worker.host) for player_name, worker in player_to_worker_mapping.items()]
        )
        config = tfe.RemoteConfig(hostmap)

        return config, player_to_worker_mapping
