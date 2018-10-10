venv: venv/bin/activate

venv/bin/activate: requirements.txt
	test -e venv/bin/activate || python -m venv venv
	. venv/bin/activate; pip install -Ur requirements.txt
	touch venv/bin/activate

install_hooks: venv
	venv/bin/pre-commit install

notebook: venv
	. venv/bin/activate; python setup.py install; python -m ipykernel install --user --name=pysyft; jupyter notebook
clean:
	rm -rf venv
