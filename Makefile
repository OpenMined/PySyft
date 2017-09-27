.PHONY: test run local

test:
	pytest

image = openmined/pysyft
run:
	docker run --rm -it -v $(PWD)/notebooks:/notebooks -w /notebooks -p 8888:8888 $(image) jupyter notebook --ip=0.0.0.0 --allow-root
