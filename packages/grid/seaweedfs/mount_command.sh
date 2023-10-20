echo "remote.configure -name=azure -type=azure -azure.account_name=redteamtest \
-azure.account_key=t7Y5enmCiG2k8o5rvItSn3Ak9tHaVTXQUTn1LQ74jQ1g5bjvs0ui/O2FXJeDaCsfI6xMPz0txtoH+AStss/Xmg==" \
        | weed shell && echo 's3.bucket.create -name azurebucket ' | weed shell && echo "remote.mount \
        -dir=/buckets/azurebucket -remote=azure/manual-test" | weed shell && weed filer.remote.sync


