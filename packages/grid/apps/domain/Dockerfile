FROM python:3.8

RUN mkdir /app
WORKDIR /app

RUN apt-get update
RUN apt-get install -y git python-dev python3-dev

RUN pip install poetry
COPY poetry.lock pyproject.toml entrypoint.sh /app/
COPY /src /app/src

WORKDIR /app/
RUN poetry export -f requirements.txt --output requirements.txt --without-hashes
RUN pip3 install -r requirements.txt

ENTRYPOINT ["sh", "entrypoint.sh"]
