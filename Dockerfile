FROM openmined/pysyft-lite as base

FROM base as builder

RUN apt-get update
RUN apt-get install -y git python3-pip python3-dev
COPY ./pip-dep /app/pip-dep

WORKDIR /app
RUN pip3 install --user -r pip-dep/requirements.txt


FROM openmined/pysyft-lite as grid_app

COPY --from=builder root/.local root/.local

COPY . /app
WORKDIR /app
ENTRYPOINT ["sh", "entrypoint.sh"]

