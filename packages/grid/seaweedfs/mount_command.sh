echo "remote.configure -name=$1 -type=azure -azure.account_name=$2 -azure.account_key=$5" | weed shell

echo "s3.bucket.create -name $3" | weed shell

echo "remote.mount -dir=/buckets/$3 -remote=$1/$4" | weed shell

weed filer.remote.sync
