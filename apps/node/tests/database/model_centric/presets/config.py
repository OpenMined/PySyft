configs = [
    (
        {
            "name": "my-federated-model",
            "version": "0.1.0",
            "batch_size": 32,
            "lr": 0.01,
            "optimizer": "SGD",
        },
        {
            "max_workers": 100,
            "pool_selection": "random",  # or "iterate"
            "num_cycles": 5,
            "do_not_reuse_workers_until_cycle": 4,
            "cycle_length": 8 * 60 * 60,  # 8 hours
            "minimum_upload_speed": 2000,  # 2 mbps
            "minimum_download_speed": 4000,  # 4 mbps
        },
    ),
    ({}, {}),
]
