![wtf](images/banner.png)

## Modes

Grid currently has two modes.

`--tree` -- experimental

Tree is the federated learning mode.  The long term goal is to have node workers store data
locally, and train public models to make them better.

`--compute`

Compute mode lets users offer compute to train models.  Data scientists can easily publish models from
a jupyter notebook and train them remotely.  Grid also offers easy utilities to try n number of
configurations concurrently.

## Running

```sh
python3.6 setup.py install
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
