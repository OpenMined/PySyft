FROM python:3.8-slim

RUN mkdir /app
WORKDIR /app

RUN apt-get update
RUN apt-get install -y git

RUN pip install poetry
COPY poetry.lock pyproject.toml entrypoint.sh /app/
COPY /src /app/src

WORKDIR /app/
RUN poetry install

ENTRYPOINT ["sh", "entrypoint.sh"]
