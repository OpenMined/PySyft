FROM python:3.6-alpine

# installing alpine packages which is needed for building python packages "especially pillow" :)
RUN apk add g++ python-dev tiff-dev zlib-dev freetype-dev libc6-compat

RUN pip install jupyter \
    && pip install http://download.pytorch.org/whl/cpu/torch-0.3.1-cp36-cp36m-linux_x86_64.whl \
    && pip install torchvision \
    && pip install msgpack \
    && rm -r /root/.cache/pip \
    && mkdir PySyft \
    && mkdir PySyft/examples

COPY syft/ /PySyft/syft/
COPY requirements.txt /PySyft/requirements.txt
COPY setup.py /PySyft/setup.py
COPY README.md /PySyft/README.md

WORKDIR /PySyft

RUN python setup.py install \
    && jupyter notebook --generate-config \
    && jupyter nbextension enable --py --sys-prefix widgetsnbextension \
    && python -m ipykernel.kernelspec \
    && mkdir /notebooks \
    && rm -rf /var/cache/apk/*

WORKDIR /notebooks

ENTRYPOINT ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--Notebook.open_browser=False", "--NotebookApp.token=''", "--allow-root"]
