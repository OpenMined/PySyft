venv: venv/bin/activate

venv/bin/activate: requirements.txt requirements_dev.txt
	test -e venv/bin/activate || virtualenv venv
	. venv/bin/activate; pip install -Ur requirements.txt; pip install -Ur requirements_dev.txt; python setup.py install
	touch venv/bin/activate

install_hooks: venv
	venv/bin/pre-commit install

notebook: venv
	(. venv/bin/activate; \
		python setup.py install; \
		python -m ipykernel install --user --name=pysyft; \
		jupyter notebook;\
	)

lab: venv
	(. venv/bin/activate; \
		python setup.py install; \
		python -m ipykernel install --user --name=pysyft; \
		jupyter lab;\
	)

.PHONY: test
test: venv
	(. venv/bin/activate; \
		python setup.py install; \
		venv/bin/coverage run setup.py test;\
		venv/bin/coverage report --fail-under 100;\
	)

.PHONY: docs
docs: venv
	(. venv/bin/activate; \
    	cd docs; \
		rm -rf ./_modules; \
		rm -rf ./_autosummary; \
		rm -rf _build; \
		sphinx-apidoc -o ./_modules ../syft; \
		make markdown; \
        cd ../; \
	)
clean:
	rm -rf venv
