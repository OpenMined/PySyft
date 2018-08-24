FROM ubuntu:18.04

RUN apt-get update \
 && apt-get install -y python3.6 \
                       python3-pip \
                       build-essential \
 && rm -rf /var/lib/apt/lists/*

RUN pip3 install jupyter \
 && pip3 install http://download.pytorch.org/whl/cpu/torch-0.3.1-cp36-cp36m-linux_x86_64.whl \
 && pip3 install torchvision \
 && rm -r /root/.cache/pip

RUN mkdir PySyft
COPY syft/ /PySyft/syft/
COPY examples/ /PySyft/examples/
COPY requirements.txt /PySyft/requirements.txt
COPY setup.py /PySyft/setup.py
COPY README.md /PySyft/README.md
WORKDIR /PySyft
RUN python3 setup.py install

RUN jupyter notebook --generate-config
RUN jupyter nbextension enable --py --sys-prefix widgetsnbextension && \
python3 -m ipykernel.kernelspec

RUN mkdir /notebooks
WORKDIR /notebooks

RUN  apt-get -y purge python3-pip \
                     build-essential \
 && apt-get -y clean

ENTRYPOINT ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--Notebook.open_browser=False", "--NotebookApp.token=''", "--allow-root"]
