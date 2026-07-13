PYTHON = poetry run python3

MAIN = main.py


all: install run

install:
	py -m poetry install

run:
	$(PYTHON) $(MAIN)

debug:
	$(PYTHON) -m pdb $(MAIN)

clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache dist

lint:
	py -m flake8 .
	py -m mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	py -m flake8 .
	py -m mypy  --strict .

.PHONY: install run debug clean lint lint-strict all
