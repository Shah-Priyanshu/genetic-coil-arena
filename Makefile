# Makefile for Genetic Coil Arena

.PHONY: help install install-dev test lint format clean benchmark run setup

# Default target
help:
	@echo "Genetic Coil Arena - Available Commands:"
	@echo ""
	@echo "Setup:"
	@echo "  setup          - Setup development environment"
	@echo "  install        - Install package in development mode"
	@echo "  install-dev    - Install with development dependencies"
	@echo ""
	@echo "Development:"
	@echo "  test           - Run test suite"
	@echo "  lint           - Run linting checks"
	@echo "  format         - Format code with black and ruff"
	@echo "  benchmark      - Run performance benchmarks"
	@echo ""
	@echo "Execution:"
	@echo "  run            - Run evolution with default settings"
	@echo "  demo           - Run quick demo (small population)"
	@echo ""
	@echo "Maintenance:"
	@echo "  clean          - Clean up build artifacts"

# Setup development environment
setup:
	python -m pip install --upgrade pip
	pip install -e ".[dev,ui,analysis]"
	pre-commit install

# Installation targets
install:
	pip install -e .

install-dev:
	pip install -e ".[dev,ui,analysis]"

# Testing
test:
	pytest tests/ -v --cov=genetic_coil_arena --cov-report=term-missing

test-fast:
	pytest tests/ -v -x -k "not slow"

# Code quality
lint:
	ruff check src/ tests/
	mypy src/genetic_coil_arena

format:
	black src/ tests/
	ruff check --fix src/ tests/

# Performance testing
benchmark:
	pytest benchmarks/ --benchmark-only --benchmark-compare

# Execution
run:
	genetic-coil evolve --generations 50 --population 128

demo:
	genetic-coil evolve --generations 10 --population 32 --output demo_results

# Utility
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

# Windows-specific versions
clean-windows:
	if exist build rmdir /s /q build
	if exist dist rmdir /s /q dist
	if exist *.egg-info rmdir /s /q *.egg-info
	if exist .pytest_cache rmdir /s /q .pytest_cache
	if exist .mypy_cache rmdir /s /q .mypy_cache
	del /s /q *.pyc 2>nul || true

# CI targets
ci-test:
	pytest tests/ --cov=genetic_coil_arena --cov-report=xml

ci-lint:
	ruff check src/ tests/ --output-format=github
	mypy src/genetic_coil_arena
