.PHONY: test run

test:
	pytest

run:
	docker run --rm -it -v $(PWD)/notebooks:/notebooks -w /notebooks -p 8888:8888 openmined/pysyft jupyter notebook --ip=0.0.0.0 --allow-root
