.PHONY: test run create-custom-image run-custom

test:
	@# Remove pyc files to avoid conflict if tests are run locally before
	@find . -name '*.pyc' -exec rm -f '{}' \;
	@docker run --rm -it -v $(PWD)/:/src/PySyft -w /src/PySyft openmined/pysyft-dev:edge pytest --flake8
run:
	docker run --rm -ti -v $(PWD)/notebooks:/notebooks -w /notebooks -p 8888:8888 openmined/pysyft-dev:edge

create-custom-image:
	docker build -f Development-Dockerfile -t pysyft-dev:local .

run-custom:
	docker run --rm -it -v $(PWD)/notebooks:/notebooks -w /notebooks -p 8888:8888 pysyft-dev:local jupyter notebook --ip=0.0.0.0 --allow-root
