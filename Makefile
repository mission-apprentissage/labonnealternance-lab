# Makefile for LBA Classifier
.PHONY: help install dev build up down logs test clean

# Default target
help: ## Show this help message
	@echo "Available commands:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Development commands (like npm scripts)
install: ## Install Python dependencies locally
	cd server && python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

dev: ## Run development server locally (requires install first)
	cd server && source .venv/bin/activate && python main.py

# Docker commands
build: ## Build Docker image
	docker-compose build

up: ## Start services with docker-compose
	docker-compose up -d

dev-up: ## Start development services with live reload
	docker-compose --profile dev up -d server-dev

down: ## Stop all services
	docker-compose down

logs: ## Show logs from all services
	docker-compose logs -f

logs-server: ## Show logs from server only
	docker-compose logs -f server

# Testing and utilities
test: ## Test the API endpoints (requires server to be running)
	@echo "Testing single classification..."
	curl -X POST http://localhost:8000/score \
		-H 'Content-Type: application/json' \
		-d '{"text": "Développeur Python recherché pour startup"}'
	@echo "\n\nTesting batch classification..."
	curl -X POST http://localhost:8000/scores \
		-H 'Content-Type: application/json' \
		-d '{"items": [{"id":"1", "text": "Développeur Python"}, {"id":"2", "text": "Stage marketing"}]}'

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