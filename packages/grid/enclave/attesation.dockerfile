ARG AZ_GUEST_LIB_VERSION="1.0.5"
ARG AZ_CLIENT_COMMIT="b613bcd"


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
    cmake . && make && cp ./AttestationClient /


# ======== [Final] Build Final Image ========== #
FROM ubuntu:22.04 as main
ARG AZ_GUEST_LIB_VERSION
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y wget && \
    wget https://packages.microsoft.com/repos/azurecore/pool/main/a/azguestattestation1/azguestattestation1_${AZ_GUEST_LIB_VERSION}_amd64.deb && \
    dpkg -i azguestattestation1_${AZ_GUEST_LIB_VERSION}_amd64.deb && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /AttestationClient /