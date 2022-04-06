FROM python:3.10.4-slim

# set UTC timezone
ENV TZ=Etc/UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN DEBIAN_FRONTEND=noninteractive \
    apt-get update && \
    apt-get install -yqq \
    git && \
    rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/root/.cache pip install --upgrade pip setuptools wheel twine

WORKDIR /hagrid
COPY ./ /hagrid

RUN python setup.py bdist_wheel
RUN twine check `find -L ./ -name "*.whl"`
RUN --mount=type=cache,target=/root/.cache pip install `find -L ./ -name "*.whl"`

# warm the cache
RUN hagrid

CMD hagrid
