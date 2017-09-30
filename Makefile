.PHONY: install test docker-build docker-run

install:
	python3 setup.py install

test: install
	pytest && pytest --flake8

notebook:
	jupyter notebook --allow-root --ip=0.0.0.0

dockerfile = Dockerfile
docker-build:
	docker build -f "$(dockerfile)" -t "pysyft" .

image = openmined/pysyft
docker-run:
	docker run -it --rm \
		-v "$(PWD)":/PySyft \
		-p 8888:8888 \
		$(image) sh
