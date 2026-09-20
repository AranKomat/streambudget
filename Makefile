.PHONY: test demo check

test:
	PYTHONPATH=src pytest -q

demo:
	PYTHONPATH=src python -m streambudget.cli demo --out runs/demo

check:
	python -m compileall -q src scripts
	PYTHONPATH=src pytest -q
