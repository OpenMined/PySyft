.PHONY: install develop test notebook \
	docker-build-base docker-build docker-build-dev docker-run \
	run-redis run-capsule run-allservices

# Platform-agnostic targets, will run locally and inside Docker,
# provided that the right dependencies are present
install:
	pip3 install -r requirements.txt
	python3 setup.py install

develop:
	pip3 install -r dev-requirements.txt
	python3 setup.py develop

test:
	@# Remove pyc files to avoid conflict if tests are run locally
	@# and inside container at the same time
	@find . -name '*.pyc' -exec rm -f '{}' \;
	pip3 install -r test-requirements.txt
	pytest && pytest --flake8

notebook:
	jupyter notebook --allow-root --ip=0.0.0.0

# Docker-related targets, to build and run a prod and a dev images
docker-build-base:
	docker build -f dockerfiles/Dockerfile.base -t pysyft-base:local .

docker-build: docker-build-base
	docker build -f dockerfiles/Dockerfile -t openmined/pysyft:local .

docker-build-dev: docker-build-base
	docker build -f dockerfiles/Dockerfile.dev -t openmined/pysyft-dev:local .

image = openmined/pysyft:local
docker-run:
	docker run -it --rm \
		-v "$(PWD)":/PySyft \
		-p 8888:8888 \
		"$(image)" sh

run-redis:
	redis-server &

run-capsule:
	FLASK_APP=/usr/bin/Capsule/capsule/local_server.py flask run &

run-allservices: run-redis run-capsule
