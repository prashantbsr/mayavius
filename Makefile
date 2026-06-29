.DEFAULT_GOAL := help

.PHONY: help setup setup-frontend setup-backend dev-frontend dev-backend test test-frontend test-backend test-e2e test-mps lint typecheck bake-corpus

help: ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

setup: setup-backend setup-frontend ## Install all deps (lightweight; no PyTorch)

setup-frontend: ## Install frontend deps
	cd frontend && npm install

setup-backend: ## Create venv (Python 3.12) + install lightweight backend deps
	cd backend && python3.12 -m venv .venv && ./.venv/bin/pip install -r requirements-dev.txt

dev-frontend: ## Run Next.js dev server (http://localhost:3000)
	cd frontend && npm run dev

dev-backend: ## Run FastAPI dev server (http://localhost:8000)
	cd backend && ./.venv/bin/uvicorn app.main:app --reload --port 8000

test: test-backend test-frontend ## Run the CI test set (excludes e2e + mps)

test-backend: ## Run backend unit + integration (no torch/MPS)
	cd backend && ./.venv/bin/python -m pytest -m "not mps and not gpu"

test-frontend: ## Run frontend unit tests + typecheck
	cd frontend && npm run test && npx tsc --noEmit

test-e2e: ## Run Playwright e2e (fixture-mode backend; no GPU)
	cd frontend && npm run test:e2e

test-mps: ## On-device MPS smoke on the 36 GB Mac (opt-in; needs requirements-ml.txt)
	cd backend && MAYAVIUS_RUN_MPS_SMOKE=1 ./.venv/bin/python -m pytest -m mps -s

bake-corpus: ## Re-bake assets/samples/*.mv4d via the real pipeline (needs ML overlay + cached weights; re-commits the blobs). HEAVY, manual, NOT CI.
	cd backend && ./.venv/bin/python scripts/bake_corpus.py

lint: ## Lint frontend
	cd frontend && npm run lint

typecheck: ## Typecheck frontend
	cd frontend && npx tsc --noEmit
