#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
[ -d .venv ] || { python3 -m venv .venv && .venv/bin/pip install -r requirements.txt; }
[ -f .env ] || cp .env.example .env
echo "Open http://127.0.0.1:8765"
.venv/bin/python -m app.main
