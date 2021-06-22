FROM python:3.9

# args and envs
ARG DEV=false
ENV DEV=$DEV
ENV LANG C.UTF-8

# pydp requires GLIBCXX_3.4.26 from gcc-9
RUN echo 'deb http://deb.debian.org/debian testing main' >> /etc/apt/sources.list
RUN apt update -y
RUN apt install -y gcc

# update pip
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install poetry

# create dirs
RUN mkdir -p /domain
RUN mkdir -p /syft
COPY ./packages/grid/apps/domain /domain
COPY ./packages/syft /syft
RUN cd /domain && poetry install
RUN cd /domain && poetry run pip install -e /syft[ci-all]

# run the domain in dev mode
CMD ["bash", "-c", "cd /domain && ./dev.sh"]


# $ cd PySyft
# $ docker build -f docker/Dockerfile . -t grid.domain
# $ docker run -it -v "`pwd`/packages/grid/apps/domain:/domain" -v "`pwd`/packages/syft:/syft" -p 5000:5000 grid.domain