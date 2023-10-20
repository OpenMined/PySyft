# docker run --entrypoint /bin/sh -p 8333:8333 -p 8888:8888 chrislusf/seaweedfs -c \
# " \
# & echo 's3.configure -access_key admin -secret_key admin -user iam -actions Read,Write,List,Tagging,Admin -apply' | \
# weed shell > /dev/null 2>&1 & weed server -s3 -s3.port=8333 -master.volumeSizeLimitMB=2048&"

export STACK_API_KEY=w9N59fxaSrb6Vl64mVHR3WVRTMZZQ7XYYTfiJ9GEUkPviQTq
echo 's3.configure -access_key admin -secret_key admin -user iam -actions Read,Write,List,Tagging,Admin -apply' | weed shell > /dev/null 2>&1 & 
weed server -s3 -s3.port=8333 -master.volumeSizeLimitMB=2048 &
flask run -p 4000 --host=0.0.0.0 


