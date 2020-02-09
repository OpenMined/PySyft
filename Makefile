venv: venv/bin/activate

REQ_DIR=pip-dep

reqs: $(REQ_DIR)/requirements.txt $(REQ_DIR)/requirements_dev.txt $(REQ_DIR)/requirements_udacity.txt $(REQ_DIR)/requirements_notebooks.txt

venv/bin/activate: reqs
	test -e venv/bin/activate || python -m venv venv
	. venv/bin/activate; pip install -Ur $(REQ_DIR)/requirements.txt; pip install -Ur $(REQ_DIR)/requirements_dev.txt; pip install -Ur $(REQ_DIR)/requirements_udacity.txt; pip install -Ur $(REQ_DIR)/requirements_notebooks.txt; python setup.py install
	touch venv/bin/activate

install_hooks: venv
	venv/bin/pre-commit install

notebook: venv reqs
	(. venv/bin/activate; \
		pip install -Ur $(REQ_DIR)/requirements_notebooks.txt; \
		python -m ipykernel install --user --name=pysyft; \
		jupyter notebook;\
	)

lab: venv
	(. venv/bin/activate; \
		python -m ipykernel install --user --name=pysyft; \
		jupyter lab;\
	)

.PHONY: test
test: venv reqs
	(. venv/bin/activate; \
		pip install -Ur $(REQ_DIR)/requirements_notebooks.txt; \
		pip install "scikit-learn>=0.21.0" "pytest" "pytest-flake8"; \
		venv/bin/coverage run -m pytest test; \
		venv/bin/coverage report -m --fail-under 95; \
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
