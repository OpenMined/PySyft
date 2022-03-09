apt-get install apt-transport-https curl gnupg
curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor > bazel.gpg
mv bazel.gpg /etc/apt/trusted.gpg.d/
echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list

apt-get update && apt-get install bazel
apt-get update && apt-get full-upgrade
apt-get update git

# change this branch with the one used for PySyTFF
git clone https://github.com/tensorflow/federated.git
cd "federated"

mkdir "/tmp/tensorflow_federated"
bazel run //tensorflow_federated/tools/python_package:build_python_package -- \
    --nightly \
    --output_dir="/tmp/tensorflow_federated"


pip install --upgrade "$(ls /tmp/tensorflow_federated/)"