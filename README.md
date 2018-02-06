# Grid

Proof of concept for python based GRID edge nodes.

## Running

```sh
python setup.py install
```

# Running with PUBSUB
### Running `ipfs_grid_worker_daemon.py`

Grid worker daemon requires an IPFS daemon to be running with pubsub support
turned on.

```sh
ipfs daemon --enable-pubsub-experiment
```

You can then run the worker daemon
```sh
python3.6 ipfs_grid_worker_daemon.py
```

Start Jupyter
```sh
jupyter notebook
```

navigate to `notebooks/pubsub/` and open `Keras Grid Client and Worker.ipynb` and
follow along in that notebook.
