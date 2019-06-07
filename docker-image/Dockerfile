FROM continuumio/miniconda3

ENV WORKSPACE /workspace

# Setup workspace environment
RUN apt-get update && apt-get install -y gcc
RUN conda install jupyter notebook
RUN pip install --no-cache-dir syft numpy

# Get the course git repo
RUN mkdir $WORKSPACE
WORKDIR $WORKSPACE
RUN git clone https://github.com/udacity/private-ai

# Make the image start the jupyer notebook
COPY ./entrypoint.sh /
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
