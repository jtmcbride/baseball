.PHONY: help setup backfill dims marts check api web dev test lint fmt clean train

SEASON_START ?= 2015
END ?= $(shell date +%F)

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup:  ## Install python + node deps
	uv sync --python 3.13
	cd apps/web && npm install

backfill:  ## Full Statcast backfill (~2-3h, resumable, safe to interrupt)
	uv run bb ingest statcast --start $(SEASON_START)-02-01 --end $(END)

refresh:  ## Re-pull the trailing window (Savant revises published data)
	uv run bb ingest refresh

dims:  ## dim_game / dim_team / dim_player + the ID crosswalk
	# MUST run after `build pitches`: dim_player is populated from the player ids
	# actually present in fact_pitch, so running it first silently builds a
	# dimension that covers only whatever was already there.
	uv run bb ingest dims
	uv run bb ingest crosswalk

build:  ## raw -> fact_pitch (+ warehouse views)
	uv run bb build pitches
	uv run bb build register

build-marts:  ## fact_pitch + dims -> marts
	uv run bb build marts

check:  ## Data quality suite (non-zero exit on any ERROR)
	uv run bb check --coverage

pipeline: backfill build dims build-marts check  ## Everything, in dependency order

api:  ## Serve the API on :8000
	uv run uvicorn bbapi.main:app --reload --host 127.0.0.1 --port 8000

web:  ## Serve the UI on :5173
	cd apps/web && npm run dev

dev:  ## API + UI together
	@$(MAKE) -j2 api web

test:  ## Python + frontend typecheck
	uv run pytest
	cd apps/web && npx tsc --noEmit

lint:
	uv run ruff check .
	uv run ruff format --check .

fmt:
	uv run ruff check --fix .
	uv run ruff format .

status:  ## Ingest progress
	uv run bb status

train:  ## Train + register both next-pitch model heads
	uv run bb-ml next-pitch
	uv run bb-ml location
	uv run bb-ml status

clean:  ## Drop derived data. Raw is preserved — never re-crawl.
	rm -rf data/lake data/db
	@echo "Removed lake + warehouse. data/raw kept; run 'make build' to rebuild."
