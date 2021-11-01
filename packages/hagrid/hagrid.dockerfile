FROM python:3.9-slim


RUN apt-get update && \
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
