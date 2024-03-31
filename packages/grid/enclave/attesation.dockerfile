ARG AZ_GUEST_LIB_VERSION="1.0.5"
ARG AZ_CLIENT_COMMIT="b613bcd"
ARG PYTHON_VERSION="3.10"
ARG NVTRUST_VERSION="1.3.0"


FROM ubuntu:22.04 as builder
ARG AZ_GUEST_LIB_VERSION
ARG AZ_CLIENT_COMMIT

# ======== [Stage 1] Install Dependencies ========== #

ENV DEBIAN_FRONTEND=noninteractive
RUN --mount=type=cache,target=/var/cache/apt/archives \
    apt update && apt upgrade -y && \
    apt-get  install -y \
    build-essential \
    libcurl4-openssl-dev \
    libjsoncpp-dev \
    libboost-all-dev \
    nlohmann-json3-dev \
    cmake \
    wget \
    git

RUN wget https://packages.microsoft.com/repos/azurecore/pool/main/a/azguestattestation1/azguestattestation1_${AZ_GUEST_LIB_VERSION}_amd64.deb && \
    dpkg -i azguestattestation1_${AZ_GUEST_LIB_VERSION}_amd64.deb

# ======== [Stage 2] Build Attestation Client ========== #

RUN git clone https://github.com/Azure/confidential-computing-cvm-guest-attestation.git && \
    cd confidential-computing-cvm-guest-attestation && \
    git checkout ${AZ_CLIENT_COMMIT} && \
    cd cvm-attestation-sample-app && \
    cmake . && make && cp ./AttestationClient /


# ======== [Final] Build Final Image ========== #
FROM python:${PYTHON_VERSION}-slim
ARG AZ_GUEST_LIB_VERSION
ARG NVTRUST_VERSION
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    wget \
    git

RUN wget https://packages.microsoft.com/repos/azurecore/pool/main/a/azguestattestation1/azguestattestation1_${AZ_GUEST_LIB_VERSION}_amd64.deb && \
    dpkg -i azguestattestation1_${AZ_GUEST_LIB_VERSION}_amd64.deb

COPY --from=builder /AttestationClient /

# Clone Nvidia nvtrust Repo
RUN git clone -b v${NVTRUST_VERSION} https://github.com/NVIDIA/nvtrust.git


# Install Nvidia Local Verifier
RUN cd nvtrust/guest_tools/gpu_verifiers/local_gpu_verifier && \
    pip install .

# Install Nvidia Attestation SDK
RUN cd nvtrust/guest_tools/attestation_sdk/dist && \
    pip install ./nv_attestation_sdk-${NVTRUST_VERSION}-py3-none-any.whl

# Cleanup
RUN rm -rf /var/lib/apt/lists/* && \
    rm -rf /nvtrust