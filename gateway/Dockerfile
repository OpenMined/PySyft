FROM python:3.7-slim
RUN apt-get update
RUN apt-get install -y git
COPY ./requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip3 install -r requirements.txt
COPY . /app
WORKDIR /app
ENTRYPOINT ["sh", "entrypoint.sh"]
