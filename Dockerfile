FROM ubuntu:18.04

RUN apt-get update \
    && apt-get install -y python3.6 \
                          python3-pip \
                          build-essential \
                          git \
    && apt-get -y autoremove \
    && apt-get -y clean  \
    && rm -rf /var/lib/apt/lists/*

COPY docker/requirements.txt /requirements.txt
RUN pip3 install -r /requirements.txt \
 && pip3 install jupyter_contrib_nbextensions \
 && pip3 install http://download.pytorch.org/whl/cpu/torch-0.3.1-cp36-cp36m-linux_x86_64.whl \
 && pip3 install torchvision \
 && rm -r /root/.cache/pip \
 && rm requirements.txt

RUN mkdir PySyft
COPY syft/ /PySyft/syft/
COPY examples/ /PySyft/examples/
COPY requirements.txt /PySyft/requirements.txt
COPY setup.py /PySyft/setup.py
COPY README.md /PySyft/README.md
WORKDIR /PySyft 
RUN python3 setup.py install

COPY docker/run_jupyter.sh /run_jupyter.sh
COPY docker/jupyter_notebook_config.py /root/.jupyter/

RUN jupyter nbextension enable --py --sys-prefix widgetsnbextension && \
python3 -m ipykernel.kernelspec

RUN mkdir /notebooks
WORKDIR /notebooks

CMD ["/run_jupyter.sh", "--allow-root"]
