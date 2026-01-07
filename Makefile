# Makefile for LBA Classifier
.PHONY: help install dev build down logs test clean

# Default target
help: ## Show this help message
	@echo "Available commands:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Development commands (like npm scripts)
install: ## Install Python dependencies locally (auto-detects macOS vs Linux)
ifeq ($(shell uname),Darwin)
	cd server && python -m venv .venv && .venv/bin/pip install -r requirements-local.txt
else
	cd server && python -m venv .venv && .venv/bin/pip install -r requirements.txt
endif

dev: ## Run development server locally with hot-reload (requires install first)
	cd server && FLASK_DEBUG=1 .venv/bin/python main.py

# Docker commands
build: ## Build Docker image
	docker-compose build

dev-up: ## Start development services with live reload (Docker)
	docker-compose --profile dev up -d server-dev

down: ## Stop all services
	docker-compose down

# Testing and utilities
test: ## Test the API endpoints (requires server to be running)
	@echo "Testing single classification..."
	curl -X POST http://localhost:8000/score \
		-H 'Content-Type: application/json' \
		-d '{"text": "Développeur Python recherché pour startup"}'
	@echo "\n\nTesting batch classification with real test data (10 job offers)..."
	curl -X POST http://localhost:8000/scores \
		-H 'Content-Type: application/json' \
		-d @test-data.json

health: ## Check API health
	curl -f http://localhost:8000/ || echo "Service not available"

version: ## Get API version
	curl -s http://localhost:8000/version | python -m json.tool

clean: ## Clean up containers and volumes
	docker-compose down -v
	docker system prune -f

# Docker build commands (similar to existing workflow)
build-prod: ## Build production image
	docker buildx build --platform linux/amd64 -t lba-classifier .

run-prod: ## Run production container
	docker run --rm -it -p 8000:8000 --name classifier lba-classifier

# Release commands
release-interactive: ## Build & Push Docker image releases (interactive)
	./.bin/mna-lab release:interactive

deploy: ## Deploy application to environment (usage: make deploy ENV=<env> USER=<username>)
	./.bin/mna-lab deploy production
