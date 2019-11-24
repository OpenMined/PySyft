FROM continuumio/miniconda3

ENV WORKSPACE /workspace

# Setup workspace environment
RUN apt-get update && apt-get install -y gcc
RUN conda install jupyter notebook
RUN pip install --no-cache-dir syft[udacity]

# Create jupyter notebook workspace
RUN mkdir $WORKSPACE
WORKDIR $WORKSPACE

# Make the image start the jupyer notebook
COPY ./entrypoint.sh /
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
