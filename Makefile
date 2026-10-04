.PHONY: test demo game-demo check

test:
	PYTHONPATH=src pytest -q

demo:
	PYTHONPATH=src python -m streambudget.cli demo --out runs/demo

game-demo:
	PYTHONPATH=src python -m streambudget.cli game demo --out runs/game-demo

check:
	python -m compileall -q src scripts
	PYTHONPATH=src pytest -q
