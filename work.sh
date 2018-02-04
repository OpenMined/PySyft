python ipfs_grid_worker_daemon.py &

while true
do
    echo "Check if pull needed"
    PULL_MSG="$(git pull)"
    if [[ $PULL_MSG != *"up-to-date"* ]]; then
        echo "Pulled, rebuilding and restarting worker"
        pkill -f ipfs_grid_worker_deamon.py
        python setup.py install
        python ipfs_grid_worker_daemon.py &  
    elif [[ $PULL_MSG = *"Aborting"* ]]; then
        echo "Not able to pull, you probably neeed to commit"
        exit 1
    fi
    sleep 5
done
