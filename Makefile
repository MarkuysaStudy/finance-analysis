PYTHON ?= python3
VENV ?= .venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python
SCRIPT ?= main.py
OUTDIR ?= outputs
OUTDIR_STRESS ?= outputs_stress
RF ?= 0.02
TCOST_BPS ?= 10
STRESS_TCOST_BPS ?= 50
TRAIN_YEARS ?= 5
TOP_K ?= 3
WEIGHTING ?= equal

.PHONY: help setup install run run-stress run-top2 run-top4 run-invvol all-runs clean dist-clean package

help:
	@echo "Available commands:"
	@echo "  make setup        Create virtual environment and install dependencies"
	@echo "  make install      Install dependencies into existing virtual environment"
	@echo "  make run          Run baseline scenario: Top-3, equal weight, 10 bps costs"
	@echo "  make run-stress   Run transaction-cost stress scenario: 50 bps costs"
	@echo "  make run-top2     Run robustness check: Top-2 selection"
	@echo "  make run-top4     Run robustness check: Top-4 selection"
	@echo "  make run-invvol   Run robustness check: Top-3 inverse-volatility weighting"
	@echo "  make all-runs     Run baseline, stress, Top-2, Top-4, and inverse-volatility scenarios"
	@echo "  make package      Create ZIP archive with script, README, Makefile, requirements, and outputs if present"
	@echo "  make clean        Remove generated output directories"
	@echo "  make dist-clean   Remove generated output directories and virtual environment"

$(VENV)/bin/python:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip

setup: $(VENV)/bin/python
	$(PIP) install -r requirements.txt

install: setup

run: setup
	$(PY) $(SCRIPT) \
		--train-years $(TRAIN_YEARS) \
		--top-k $(TOP_K) \
		--rf $(RF) \
		--tcost-bps $(TCOST_BPS) \
		--weighting $(WEIGHTING) \
		--outdir $(OUTDIR)

run-stress: setup
	$(PY) $(SCRIPT) \
		--train-years $(TRAIN_YEARS) \
		--top-k 3 \
		--rf $(RF) \
		--tcost-bps $(STRESS_TCOST_BPS) \
		--weighting equal \
		--outdir $(OUTDIR_STRESS)

run-top2: setup
	$(PY) $(SCRIPT) \
		--train-years $(TRAIN_YEARS) \
		--top-k 2 \
		--rf $(RF) \
		--tcost-bps $(TCOST_BPS) \
		--weighting equal \
		--outdir outputs_top2

run-top4: setup
	$(PY) $(SCRIPT) \
		--train-years $(TRAIN_YEARS) \
		--top-k 4 \
		--rf $(RF) \
		--tcost-bps $(TCOST_BPS) \
		--weighting equal \
		--outdir outputs_top4

run-invvol: setup
	$(PY) $(SCRIPT) \
		--train-years $(TRAIN_YEARS) \
		--top-k 3 \
		--rf $(RF) \
		--tcost-bps $(TCOST_BPS) \
		--weighting invvol \
		--outdir outputs_invvol

all-runs: run run-stress run-top2 run-top4 run-invvol

clean:
	rm -rf outputs \
		outputs_stress \
		outputs_top2 \
		outputs_top4 \
		outputs_invvol
	rm -f ml_asset_management_reproducibility_package.zip

dist-clean: clean
	rm -rf $(VENV)

package:
	zip -r ml_asset_management_reproducibility_package.zip \
		$(SCRIPT) README.md Makefile requirements.txt \
		outputs \
		outputs_stress \
		outputs_top2 \
		outputs_top4 \
		outputs_invvol \
		-x "*/.DS_Store" || true
