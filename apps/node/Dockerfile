FROM python:3.7-slim
RUN mkdir /app
WORKDIR /app

RUN apt-get update
RUN apt-get install -y git python3-pip

RUN pip3 install poetry
COPY poetry.lock pyproject.toml entrypoint.sh /app/
COPY /src /app/src

WORKDIR /app/
RUN poetry install

ENTRYPOINT ["sh", "entrypoint.sh"]
