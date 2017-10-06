.PHONY: test run local

test:
	pytest && pytest --flake8
run:
	docker run --rm -it -v $(PWD)/notebooks:/notebooks -w /notebooks -p 8888:8888 openmined/pysyft-dev:edge jupyter notebook --ip=0.0.0.0 --allow-root
custom:
	docker run --rm -it -v $(PWD)/notebooks:/notebooks -w /notebooks -p 8888:8888 $(docker) jupyter notebook --ip=0.0.0.0 --allow-root
