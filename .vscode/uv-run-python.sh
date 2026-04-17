#!/bin/bash
source .venv/bin/activate
exec uv run python "$@"
