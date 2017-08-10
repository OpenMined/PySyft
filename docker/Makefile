help:
	@cat Makefile

DOCKERFILE=Dockerfile

build:
	docker build -t pysyft -f $(DOCKERFILE) .

bash: build
	docker run -it pysyft bash

ipython: build
	docker run -it pysyft ipython

notebook: build
	docker run -it -p 8888:8888 pysyft

test: build
	docker run -it pysyft pytest PySyft/tests/test_linear_classifier.py
