FROM python:3.9-slim

RUN apt-get update && apt-get install -y git
RUN pip install --upgrade pip setuptools wheel twine

WORKDIR /hagrid
COPY ./ /hagrid

RUN python setup.py bdist_wheel
RUN twine check `find -L ./ -name "*.whl"`
RUN pip install `find -L ./ -name "*.whl"`

# warm the cache
RUN hagrid

CMD hagrid