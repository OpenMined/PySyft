FROM python:3.10.4-slim as build

WORKDIR /hagrid
COPY ./ /hagrid

RUN pip install --upgrade pip setuptools wheel twine
RUN python setup.py bdist_wheel
RUN twine check `find -L ./dist -name "*.whl"`

FROM python:3.10.4-slim as backend

# set UTC timezone
ENV TZ=Etc/UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN DEBIAN_FRONTEND=noninteractive \
    apt-get update && \
    apt-get install -yqq \
    git && \
    rm -rf /var/lib/apt/lists/*

COPY --from=build /hagrid/dist /hagrid
RUN pip install `find -L /hagrid -name "*.whl"`

# warm the cache
RUN hagrid
CMD hagrid
