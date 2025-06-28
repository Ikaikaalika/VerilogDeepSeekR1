# DeepSeek R1 Verilog Fine-tuning Pipeline Makefile

.PHONY: help install install-dev clean lint format test test-cov setup-data train-sft train-ppo evaluate deploy

# Default target
help:
	@echo "Available commands:"
	@echo "  install      - Install package and dependencies"
	@echo "  install-dev  - Install package with development dependencies"
	@echo "  clean        - Clean build artifacts and caches"
	@echo "  lint         - Run code linting (flake8, mypy)"
	@echo "  format       - Format code (black, isort)"
	@echo "  test         - Run tests"
	@echo "  test-cov     - Run tests with coverage"
	@echo "  setup-data   - Download and preprocess datasets"
	@echo "  train-sft    - Run supervised fine-tuning"
	@echo "  train-ppo    - Run PPO training"
	@echo "  evaluate     - Evaluate trained model"
	@echo "  deploy       - Deploy model for production"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e .[dev]
	pre-commit install

# Cleaning
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/

# Code quality
lint:
	flake8 src/ scripts/
	mypy src/

format:
	black src/ scripts/
	isort src/ scripts/

format-check:
	black --check src/ scripts/
	isort --check-only src/ scripts/

# Testing
test:
	pytest tests/

test-cov:
	pytest tests/ --cov=src --cov-report=html --cov-report=term

# Data pipeline
setup-data:
	python scripts/download_and_preprocess.py

# Training pipeline
train-sft:
	python scripts/train_supervised.py \
		--model_name deepseek-ai/deepseek-r1-distill-qwen-32b \
		--data_dir data/processed \
		--output_dir checkpoints/deepseek-verilog-sft \
		--num_train_epochs 3

train-ppo:
	python scripts/train_ppo.py \
		--model_path checkpoints/deepseek-verilog-sft \
		--data_dir data/processed \
		--output_dir checkpoints/deepseek-verilog-ppo \
		--num_ppo_steps 1000

train-thunder:
	bash scripts/train_thunder.sh --phase both

# Evaluation and deployment
evaluate:
	python scripts/evaluate_model.py \
		--model_path checkpoints/deepseek-verilog-ppo \
		--data_dir data/processed \
		--output_dir evaluation_results

deploy:
	python scripts/deploy_model.py \
		--model_path checkpoints/deepseek-verilog-ppo \
		--output_dir deployed_model \
		--is_peft_model \
		--merge_lora \
		--optimize_for_inference

# Monitoring
monitor:
	python scripts/monitor_training.py \
		--checkpoint_dir checkpoints \
		--log_dir logs

# Docker support
docker-build:
	docker build -t deepseek-verilog:latest .

docker-run:
	docker run --gpus all -v $(PWD):/workspace deepseek-verilog:latest

# Environment setup
setup-env:
	bash scripts/install_dependencies.sh

# Quick development setup
dev-setup: install-dev setup-data
	@echo "Development environment ready!"

# CI/CD pipeline simulation
ci: format-check lint test
	@echo "CI pipeline completed successfully!"

# Full pipeline (for testing)
full-pipeline: setup-data train-sft train-ppo evaluate deploy
	@echo "Full training pipeline completed!"