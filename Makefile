venv: venv/bin/activate

venv/bin/activate: requirements.txt
	test -e venv/bin/activate || python -m venv venv
	. venv/bin/activate; pip install -Ur requirements.txt; python setup.py install
	touch venv/bin/activate

install_hooks: venv
	venv/bin/pre-commit install

notebook: venv
	(. venv/bin/activate; \
		python setup.py install; \
		python -m ipykernel install --user --name=pysyft; \
		jupyter notebook;\
	)
.PHONY: test
test: venv
	(. venv/bin/activate; \
		python setup.py install; \
		venv/bin/coverage run setup.py test;\
		venv/bin/coverage report --fail-under 100;\
	)

docs: venv
	(. venv/bin/activate; \
    	cd docs; \
		rm -rf ./_modules; \
		rm -rf ./_autosummary; \
		rm -rf _build; \
		sphinx-apidoc -o ./_modules ../syft; \
		make html; \
        cd ../; \
	)
clean:
	rm -rf venv
