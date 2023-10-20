FROM chrislusf/seaweedfs:latest

WORKDIR /

RUN apk update && apk upgrade --available
RUN apk add --no-cache python3 py3-pip ca-certificates bash 

COPY ./requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r requirements.txt


# RUN pip install flask==2.3.2 --root-user-action=ignore 
# RUN pip install flask_shell2http==1.9.1 --root-user-action=ignore 

COPY ./new_seaweed.sh /new_seaweed.sh
COPY ./app.py /app.py
COPY ./command.sh /command.sh


RUN chmod +x /new_seaweed.sh
RUN chmod +x /command.sh


ENTRYPOINT ["bash", "new_seaweed.sh"]
# CMD ["sh", '/seaweed.sh']
