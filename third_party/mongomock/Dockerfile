# OS+dist should be kept in sync with .travis.yml
FROM ubuntu:focal

RUN apt-get update && apt-get install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev git netcat
RUN curl -L https://raw.githubusercontent.com/yyuu/pyenv-installer/master/bin/pyenv-installer | bash

ENV PYENV_ROOT /root/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH

RUN eval "$(pyenv init -)" && eval "$(pyenv virtualenv-init -)"
RUN pyenv install 3.6.15
RUN pyenv install 3.7.9
RUN pyenv install 3.8.12
RUN pyenv install 3.9.10
RUN pyenv install pypy3.8-7.3.7
RUN pyenv install 3.10.2
RUN pyenv local 3.6.15 3.7.9 3.8.12 3.9.10 3.10.2 pypy3.8-7.3.7

RUN pip install tox
