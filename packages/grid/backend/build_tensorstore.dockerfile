FROM python:3.10.7-slim as build
RUN apt-get -y update --allow-insecure-repositories
RUN apt-get -y upgrade
RUN apt-get -y dist-upgrade
RUN apt-get -y install git wget gcc g++ curl make sudo
RUN git clone https://github.com/google/tensorstore /tf_store
WORKDIR /tf_store
RUN git checkout tags/v0.1.25 -b v0.1.25
RUN python -m pip install -U pip setuptools wheel
RUN export BAZEL_VERSION=5.1.0 && wget -O bazel "https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-linux-arm64" && chmod +x bazel
RUN mv bazelisk-linux-arm64 bazel
RUN chmod +x bazel
RUN export PATH=`pwd`:$PATH
RUN pip install -r third_party/pypa/cibuildwheel_requirements_frozen.txt
RUN cp .bazelrc /root/ci_bazelrc
RUN apt-get install libavif-dev

# use bazel 5.1

# ignore bazelisk because 5.0 doesnt work on aarch64

# replace perl third party rules with arm64 version
# def repo():
#     maybe(
#         third_party_http_archive,
#         name = "rules_perl",
#         urls = [
#             "https://github.com/bazelbuild/rules_perl/archive/022b8daf2bb4836ac7a50e4a1d8ea056a3e1e403.tar.gz",
#         ],
#         sha256 = "7d4e17a5850446388ab74a3d884d80731d45931aa6ac93edb9efbd500628fdcb",
#         strip_prefix = "rules_perl-022b8daf2bb4836ac7a50e4a1d8ea056a3e1e403",
#     )

# change setup.py to use bazel (not bazelisk python)

# manually patch version string in setup.py
# version="0.1.25"
# comment out use_scm_version

# run
# export TENSORSTORE_SYSTEM_LIBS=org_aomedia_avif
# python -m pip wheel ./ --wheel-dir=./ --no-deps -v
