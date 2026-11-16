SHELL := /bin/bash
COMPOSE := docker compose -f infra/docker-compose.yml

.DEFAULT_GOAL := help

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

build: ## build the pipeline (all extra) and web images
	$(COMPOSE) build

up: ## start db + api + web
	$(COMPOSE) up -d --build

down: ## stop everything (keeps volumes)
	$(COMPOSE) down

reset: ## DESTROY db + snapshots, then start fresh
	$(COMPOSE) down -v
	$(COMPOSE) up -d --build

discovery: ## start with SearXNG enabled
	$(COMPOSE) --profile discovery up -d --build

migrate: ## apply SQL migrations
	$(COMPOSE) run --rm api python -m fmd.cli migrate

seed: ## load config registry into the DB
	$(COMPOSE) run --rm api python -m fmd.cli seed

crawl: ## run the frontier within configured budgets (ARGS=--dry-run --limit 5)
	$(COMPOSE) run --rm api python -m fmd.cli crawl $(ARGS)

extract: ## snapshots -> extractions -> claims (ARGS=--limit 5)
	$(COMPOSE) run --rm api python -m fmd.cli extract $(ARGS)

score: ## claims -> scores + insight sets (ARGS=--dry-run)
	$(COMPOSE) run --rm api python -m fmd.cli score $(ARGS)

eval: ## golden set, provenance gate, drift (offline; no DB, no network)
	cd packages/pipeline && uv run python -m fmd.cli eval --golden ../../data/golden --report-out ../../reports/eval.json

dev: ## local dev (uvicorn + next dev, no docker for the apps)
	cd packages/pipeline && uv run uvicorn fmd.api:app --reload --port 8000 & \
	cd apps/web && pnpm dev

test: ## unit + integration, mock provider, no network
	cd packages/pipeline && uv sync --all-extras --quiet && uv run pytest -q
	cd apps/web && pnpm test

lint: ## ruff + eslint
	cd packages/pipeline && uv sync --all-extras --quiet && uv run ruff check . && uv run ruff format --check .
	cd apps/web && pnpm lint

fmt: ## auto-format
	cd packages/pipeline && uv sync --all-extras --quiet && uv run ruff format . && uv run ruff check --fix .
	cd apps/web && pnpm format

.PHONY: help build up down reset discovery migrate seed crawl extract score eval dev test lint fmt
