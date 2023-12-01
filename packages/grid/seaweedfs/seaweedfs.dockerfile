FROM chrislusf/seaweedfs:3.57

WORKDIR /

RUN apk update && apk upgrade --available
RUN apk add --no-cache python3 py3-pip ca-certificates bash

COPY ./requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY ./start.sh /start.sh
COPY ./mount_command.sh /mount_command.sh
COPY ./app.py /app.py

RUN chmod +x /start.sh
RUN chmod +x /mount_command.sh

CMD ["bash", "/start.sh"]
