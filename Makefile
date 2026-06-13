.DEFAULT_GOAL := help

.PHONY: help setup setup-frontend setup-backend dev-frontend dev-backend test test-frontend test-backend lint typecheck

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

setup: setup-backend setup-frontend ## Install all deps (lightweight; no PyTorch)

setup-frontend: ## Install frontend deps
	cd frontend && npm install

setup-backend: ## Create venv + install lightweight backend deps
	cd backend && python3 -m venv .venv && ./.venv/bin/pip install -r requirements-dev.txt

dev-frontend: ## Run Next.js dev server (http://localhost:3000)
	cd frontend && npm run dev

dev-backend: ## Run FastAPI dev server (http://localhost:8000)
	cd backend && ./.venv/bin/uvicorn app.main:app --reload --port 8000

test: test-backend test-frontend ## Run all tests/checks

test-backend: ## Run backend smoke/unit tests
	cd backend && ./.venv/bin/python -m pytest

test-frontend: ## Typecheck frontend (no test runner wired yet)
	cd frontend && npx tsc --noEmit

lint: ## Lint frontend
	cd frontend && npm run lint

typecheck: ## Typecheck frontend
	cd frontend && npx tsc --noEmit
