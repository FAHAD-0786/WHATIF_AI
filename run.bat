@echo off
if not exist .venv ( python -m venv .venv && .venv\Scripts\pip install -r requirements.txt )
if not exist .env copy .env.example .env
.venv\Scripts\python -m app.main
