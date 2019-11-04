venv: venv/bin/activate

venv/bin/activate: requirements.txt
	test -e venv/bin/activate || virtualenv venv
	. venv/bin/activate; pip install -Ur requirements.txt; python setup.py install
	touch venv/bin/activate

install_hooks: venv
	venv/bin/pre-commit install

notebook: venv
	(. venv/bin/activate; \
		python setup.py install; \
		python -m ipykernel install --user --name=grid; \
		jupyter notebook;\
	)

lab: venv
	(. venv/bin/activate; \
		python setup.py install; \
		python -m ipykernel install --user --name=grid; \
		jupyter lab;\
	)

.PHONY: test
test: venv
	(. venv/bin/activate; \
		python setup.py install; \
		venv/bin/coverage run setup.py test;\
	)

clean:
	rm -rf venv